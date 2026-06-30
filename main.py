import os
import base64
from datetime import datetime, date, timedelta

import face_recognition
import numpy as np
import plotly.graph_objects as go
from fastapi import FastAPI, Request, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import Person, FaceEncoding, Attendance,Holiday
from face_utils import get_face_encoding, string_to_encoding

from zoneinfo import ZoneInfo

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
MIN_OUT_TIME = timedelta(minutes=10)


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

def check_duplicate_face(new_encodings, db: Session, tolerance=0.45):
    all_encodings = db.query(FaceEncoding).all()

    for new_encoding in new_encodings:
        for stored in all_encodings:
            stored_encoding = string_to_encoding(stored.encoding)

            distance = face_recognition.face_distance(
                [stored_encoding],
                new_encoding
            )[0]

            if distance <= tolerance:
                return stored.person, distance

    return None, None

@app.post("/register")
def register_person(
    person_id: str = Form(...),
    full_name: str = Form(...),
    email: str = Form(...),
    contact: str = Form(...),
    college: str = Form(None),
    semester: str = Form(None),
    department: str = Form(...),
    straight_image: str = Form(...),
    left_image: str = Form(...),
    right_image: str = Form(...),
    db: Session = Depends(get_db)
):
    existing_person = db.query(Person).filter(Person.person_id == person_id).first()

    if existing_person:
        raise HTTPException(
            status_code=400,
            detail="Person ID already exists"
        )

    images = {
        "straight": straight_image,
        "left": left_image,
        "right": right_image
    }

    image_paths = {}
    new_encodings = {}
    temp_encoding_arrays = []

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

            encoding_string, message = get_face_encoding(image_path)

            if message != "success":
                for path in image_paths.values():
                    if os.path.exists(path):
                        os.remove(path)

                if os.path.exists(image_path):
                    os.remove(image_path)

                raise HTTPException(
                    status_code=400,
                    detail=f"{pose_name} image error: {message}"
                )

            encoding_array = string_to_encoding(encoding_string)

            image_paths[pose_name] = image_path
            new_encodings[pose_name] = encoding_string
            temp_encoding_arrays.append(encoding_array)

        duplicate_person, distance = check_duplicate_face(
            temp_encoding_arrays,
            db
        )

        if duplicate_person:
            for path in image_paths.values():
                if os.path.exists(path):
                    os.remove(path)

            raise HTTPException(
                status_code=400,
                detail=f"This face is already registered as {duplicate_person.full_name} ({duplicate_person.person_id})"
            )

        new_person = Person(
            person_id=person_id,
            full_name=full_name,
            email=email,
            contact=contact,
            college=college,
            semester=semester,
            department=department
        )

        db.add(new_person)
        db.commit()
        db.refresh(new_person)

        for pose_name, encoding_string in new_encodings.items():
            face_encoding = FaceEncoding(
                person_id_fk=new_person.id,
                pose_name=pose_name,
                image_path=image_paths[pose_name],
                encoding=encoding_string
            )

            db.add(face_encoding)

        db.commit()

    except HTTPException:
        raise

    except Exception as e:
        for path in image_paths.values():
            if os.path.exists(path):
                os.remove(path)

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
            return {"status": "waiting", "message": "No face detected"}

        if len(face_locations) > 1:
            return {"status": "multiple", "message": "Only one face should be visible"}

        current_encodings = face_recognition.face_encodings(image, face_locations)

        if len(current_encodings) == 0:
            return {"status": "error", "message": "Face encoding failed"}

        current_encoding = current_encodings[0]

        all_encodings = db.query(FaceEncoding).all()

        if not all_encodings:
            return {"status": "empty", "message": "No registered persons found"}

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
            return {"status": "unknown", "message": "Unknown person"}

        ist_now = datetime.now(ZoneInfo("Asia/Kolkata"))
        today = ist_now.date()
        current_time = ist_now.time()

        existing_attendance = db.query(Attendance).filter(
            Attendance.person_id_fk == best_match_person.id,
            Attendance.date == today
        ).first()

        if not existing_attendance:
            attendance = Attendance(
                person_id_fk=best_match_person.id,
                date=today,
                in_time=current_time,
                out_time=None,
                status="Present"
            )

            db.add(attendance)
            db.commit()

            return {
                "status": "in_marked",
                "message": "In Time marked successfully",
                "person_id": best_match_person.person_id,
                "full_name": best_match_person.full_name,
                "department": best_match_person.department,
                "date": str(today),
                "in_time": str(current_time),
                "out_time": "-"
            }

        if existing_attendance.out_time is None:
            in_datetime = datetime.combine(
                today,
                existing_attendance.in_time
            )

            current_datetime = datetime.combine(
                today,
                current_time
            )

            time_difference = current_datetime - in_datetime

            if time_difference >= MIN_OUT_TIME:
                existing_attendance.out_time = current_time
                db.commit()

                return {
                    "status": "out_marked",
                    "message": "Out Time marked successfully",
                    "person_id": best_match_person.person_id,
                    "full_name": best_match_person.full_name,
                    "department": best_match_person.department,
                    "date": str(existing_attendance.date),
                    "in_time": str(existing_attendance.in_time),
                    "out_time": str(existing_attendance.out_time)
                }

            return {
                "status": "already_in",
                "message": "In Time already marked. Out Time allowed after 10 min.",
                "person_id": best_match_person.person_id,
                "full_name": best_match_person.full_name,
                "department": best_match_person.department,
                "date": str(existing_attendance.date),
                "in_time": str(existing_attendance.in_time),
                "out_time": "-"
            }

        return {
            "status": "completed",
            "message": "In Time and Out Time already marked today",
            "person_id": best_match_person.person_id,
            "full_name": best_match_person.full_name,
            "department": best_match_person.department,
            "date": str(existing_attendance.date),
            "in_time": existing_attendance.in_time.strftime("%H:%M:%S"),
            "out_time": existing_attendance.out_time.strftime("%H:%M:%S")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    recent_attendance = db.query(Attendance).order_by(Attendance.id.desc()).limit(5).all()

    return templates.TemplateResponse(
    request,
    "dashboard.html",
    {
        "total_registered": total_registered,
        "today_present": today_present,
        "today_absent": today_absent,
        "recent_attendance": recent_attendance
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

    holidays = db.query(Holiday).all()

    while current_date <= end_date:

        attendance = next(
            (
                a for a in attendances
                if a.date == current_date
            ),
            None
        )

        holiday_found = None

        for holiday in holidays:
            if holiday.start_date <= current_date <= holiday.end_date:
                holiday_found = holiday
                break

        if attendance:

            status = "Present"
            in_time = attendance.in_time
            out_time = attendance.out_time

        elif current_date.weekday() == 6:

            status = "Sunday Holiday"
            in_time = None
            out_time = None

        elif holiday_found:

            status = holiday_found.holiday_name
            in_time = None
            out_time = None

        else:

            status = "Absent"
            in_time = None
            out_time = None

        report.append({

            "date": current_date,
            "day": current_date.strftime("%A"),
            "status": status,
            "in_time": in_time,
            "out_time": out_time

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

@app.get("/get-graph-data")
def get_graph_data(db: Session = Depends(get_db)):
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()

    labels = []
    present_values = []

    for i in range(6, -1, -1):
        day = today - timedelta(days=i)

        present_count = db.query(Attendance).filter(
            Attendance.date == day
        ).count()

        labels.append(day.strftime("%d %b"))
        present_values.append(present_count)

    return {
        "labels": labels,
        "values": present_values
    }

@app.get("/holidays-page", response_class=HTMLResponse)
def holidays_page(request: Request, db: Session = Depends(get_db)):
    holidays = db.query(Holiday).order_by(Holiday.start_date.desc()).all()

    return templates.TemplateResponse(
        request,
        "holidays.html",
        {
            "holidays": holidays
        }
    )


@app.post("/add-holiday")
def add_holiday(
    holiday_name: str = Form(...),
    holiday_type: str = Form(...),
    start_date: str = Form(...),
    end_date: str = Form(...),
    db: Session = Depends(get_db)
):
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()

    if end_date_obj < start_date_obj:
        raise HTTPException(
            status_code=400,
            detail="End date cannot be before start date"
        )

    holiday = Holiday(
        holiday_name=holiday_name,
        holiday_type=holiday_type,
        start_date=start_date_obj,
        end_date=end_date_obj
    )

    db.add(holiday)
    db.commit()

    return {
        "message": "Holiday added successfully"
    }


@app.post("/delete-holiday/{holiday_id}")
def delete_holiday(
    holiday_id: int,
    db: Session = Depends(get_db)
):
    holiday = db.query(Holiday).filter(Holiday.id == holiday_id).first()

    if not holiday:
        raise HTTPException(status_code=404, detail="Holiday not found")

    db.delete(holiday)
    db.commit()

    return {
        "message": "Holiday deleted successfully"
    }

@app.post("/check-face-position")
def check_face_position(image_data: str = Form(...)):
    temp_path = os.path.join(UPLOAD_DIR, "temp_check.jpg")

    try:
        image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        with open(temp_path, "wb") as file:
            file.write(image_bytes)

        image = face_recognition.load_image_file(temp_path)
        face_locations = face_recognition.face_locations(image)

        height, width, _ = image.shape

        if len(face_locations) == 0:
            return {"ready": False, "message": "No face detected"}

        if len(face_locations) > 1:
            return {"ready": False, "message": "Only one face allowed"}

        top, right, bottom, left = face_locations[0]

        face_width = right - left

        if face_width < width * 0.15:
            return {"ready": False, "message": "Move closer"}

        if face_width > width * 0.60:
            return {"ready": False, "message": "Move back"}

        return {
            "ready": True,
            "message": "Face detected",
            "box": {
                "top": top,
                "right": right,
                "bottom": bottom,
                "left": left,
                "image_width": width,
                "image_height": height
            }
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/next-person-id")
def next_person_id(db: Session = Depends(get_db)):
    last_person = db.query(Person).order_by(Person.id.desc()).first()

    if not last_person:
        return {"person_id": "P001"}

    next_number = last_person.id + 1
    next_id = f"P{next_number:03d}"

    return {"person_id": next_id}