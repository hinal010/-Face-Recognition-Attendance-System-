from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# DATABASE_URL = "postgresql://postgres:Hinal%402554@localhost:5432/attendance_db"
DATABASE_URL = "postgresql://postgres:MujYeaLljpTFGmcwXmPrQBqNSXZSPuYt@postgres.railway.internal:5432/railway"
engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()