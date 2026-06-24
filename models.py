from sqlalchemy import Column, Integer, String, Text, Date, Time, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base
from zoneinfo import ZoneInfo


class Person(Base):
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(String, unique=True, nullable=False, index=True)
    full_name = Column(String, nullable=False)
    department = Column(String, nullable=False)
    created_at = Column(
    DateTime,
    default=lambda: datetime.now(
        ZoneInfo("Asia/Kolkata")
    )
)

    encodings = relationship(
        "FaceEncoding",
        back_populates="person",
        cascade="all, delete"
    )

    attendances = relationship(
        "Attendance",
        back_populates="person",
        cascade="all, delete"
    )


class FaceEncoding(Base):
    __tablename__ = "face_encodings"

    id = Column(Integer, primary_key=True, index=True)
    person_id_fk = Column(Integer, ForeignKey("persons.id"), nullable=False)
    pose_name = Column(String, nullable=False)
    image_path = Column(String, nullable=False)
    encoding = Column(Text, nullable=False)
    created_at = Column(
    DateTime,
    default=lambda: datetime.now(
        ZoneInfo("Asia/Kolkata")
    )
)

    person = relationship("Person", back_populates="encodings")


class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, index=True)
    person_id_fk = Column(Integer, ForeignKey("persons.id"), nullable=False)
    date = Column(Date, nullable=False)
    time = Column(Time, nullable=False)
    status = Column(String, default="Present")

    person = relationship("Person", back_populates="attendances")