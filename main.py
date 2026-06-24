import os
import base64
from datetime import datetime, date, timedelta

import face_recognition
import numpy as np

from fastapi import FastAPI, Request, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import Person, FaceEncoding, Attendance
from face_utils import get_face_encoding, string_to_encoding

from zoneinfo import ZoneInfo

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)



@app.get("/register-page", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse(
        request,
        "register.html"
    )


@app.get("/attendance-page", response_class=HTMLResponse)
def attendance_page(request: Request):
    return templates.TemplateResponse(
        request,
        "attendance.html"
    )


@app.post("/register")
def register_person(
    person_id: str = Form(...),
    full_name: str = Form(...),
    department: str = Form(...),
    straight_image: str = Form(...),
    left_image: str = Form(...),
    right_image: str = Form(...),
    db: Session = Depends(get_db)
):
    existing_person = db.query(Person).filter(Person.person_id == person_id).first()

    if existing_person:
        raise HTTPException(status_code=400, detail="Person ID already exists")

    new_person = Person(
        person_id=person_id,
        full_name=full_name,
        department=department
    )

    db.add(new_person)
    db.commit()
    db.refresh(new_person)

    images = {
        "straight": straight_image,
        "left": left_image,
        "right": right_image
    }

    try:
        for pose_name, image_data in images.items():
            image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)

            image_path = os.path.join(
                UPLOAD_DIR,
                f"{person_id}_{pose_name}.jpg"
            )

            with open(image_path, "wb") as file:
                file.write(image_bytes)

            encoding, message = get_face_encoding(image_path)

            if message != "success":
                db.delete(new_person)
                db.commit()

                if os.path.exists(image_path):
                    os.remove(image_path)

                raise HTTPException(
                    status_code=400,
                    detail=f"{pose_name} image error: {message}"
                )

            face_encoding = FaceEncoding(
                person_id_fk=new_person.id,
                pose_name=pose_name,
                image_path=image_path,
                encoding=encoding
            )

            db.add(face_encoding)

        db.commit()

    except HTTPException:
        raise

    except Exception as e:
        db.delete(new_person)
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message": "Registration successful",
        "person_id": person_id,
        "full_name": full_name,
        "department": department
    }


@app.post("/mark-attendance")
def mark_attendance(
    image_data: str = Form(...),
    db: Session = Depends(get_db)
):
    temp_path = os.path.join(UPLOAD_DIR, "temp_attendance.jpg")

    try:
        image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        with open(temp_path, "wb") as file:
            file.write(image_bytes)

        image = face_recognition.load_image_file(temp_path)
        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 0:
            return {
                "status": "waiting",
                "message": "No face detected"
            }

        if len(face_locations) > 1:
            return {
                "status": "multiple",
                "message": "Only one face should be visible"
            }

        current_encodings = face_recognition.face_encodings(image, face_locations)

        if len(current_encodings) == 0:
            return {
                "status": "error",
                "message": "Face encoding failed"
            }

        current_encoding = current_encodings[0]

        all_encodings = db.query(FaceEncoding).all()

        if not all_encodings:
            return {
                "status": "empty",
                "message": "No registered persons found"
            }

        best_match_person = None
        best_distance = 1.0

        for stored in all_encodings:
            stored_encoding = string_to_encoding(stored.encoding)

            distance = face_recognition.face_distance(
                [stored_encoding],
                current_encoding
            )[0]

            if distance < best_distance:
                best_distance = distance
                best_match_person = stored.person

        if best_distance > 0.45:
            return {
                "status": "unknown",
                "message": "Unknown person"
            }

        ist_now = datetime.now(ZoneInfo("Asia/Kolkata"))
        today = ist_now.date()
        current_time = ist_now.time()

        existing_attendance = db.query(Attendance).filter(
            Attendance.person_id_fk == best_match_person.id,
            Attendance.date == today
        ).first()

        if existing_attendance:
            return {
                "status": "already_marked",
                "message": "Attendance already marked today",
                "person_id": best_match_person.person_id,
                "full_name": best_match_person.full_name,
                "department": best_match_person.department,
                "date": str(existing_attendance.date),
                "time": str(existing_attendance.time)
            }

        attendance = Attendance(
            person_id_fk=best_match_person.id,
            date=today,
            time=current_time,
            status="Present"
        )

        db.add(attendance)
        db.commit()

        return {
            "status": "marked",
            "message": "Attendance marked successfully",
            "person_id": best_match_person.person_id,
            "full_name": best_match_person.full_name,
            "department": best_match_person.department,
            "date": str(today),
            "time": str(current_time)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    total_registered = db.query(Person).count()

    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()

    today_present = db.query(Attendance).filter(
        Attendance.date == today
    ).count()

    today_absent = total_registered - today_present

    return templates.TemplateResponse(
    request,
    "dashboard.html",
    {
        "total_registered": total_registered,
        "today_present": today_present,
        "today_absent": today_absent
    }
)

@app.get("/persons-page", response_class=HTMLResponse)
def persons_page(request: Request, db: Session = Depends(get_db)):
    persons = db.query(Person).order_by(Person.id.desc()).all()

    return templates.TemplateResponse(
        request,
        "persons.html",
        {
            "persons": persons
        }
    )


@app.get("/person-attendance/{person_id}", response_class=HTMLResponse)
def person_attendance(
    person_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    person = db.query(Person).filter(Person.id == person_id).first()

    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    attendances = db.query(Attendance).filter(
        Attendance.person_id_fk == person.id
    ).all()

    present_dates = {
        attendance.date for attendance in attendances
    }

    start_date = person.created_at.date()

    end_date = datetime.now(
        ZoneInfo("Asia/Kolkata")
    ).date()

    report = []
    current_date = start_date

    while current_date <= end_date:
        if current_date in present_dates:
            status = "Present"

        elif current_date.weekday() == 6:
            status = "Sunday Holiday"

        else:
            status = "Absent"

        report.append({
            "date": current_date,
            "day": current_date.strftime("%A"),
            "status": status
        })

        current_date += timedelta(days=1)

    return templates.TemplateResponse(
        request,
        "person_attendance.html",
        {
            "person": person,
            "report": report
        }
    )

@app.post("/delete-person/{person_id}")
def delete_person(
    person_id: int,
    db: Session = Depends(get_db)
):
    person = db.query(Person).filter(Person.id == person_id).first()

    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    db.delete(person)
    db.commit()

    return {
        "message": "Person deleted successfully"
    }