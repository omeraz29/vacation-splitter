from __future__ import annotations

import os
import secrets
from datetime import datetime
from typing import List, Optional, Dict, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Field, Session, create_engine, select


# -----------------------------
# Database models
# -----------------------------
class Trip(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    code: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Member(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    trip_id: int = Field(foreign_key="trip.id", index=True)
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Expense(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    trip_id: int = Field(foreign_key="trip.id", index=True)
    description: str
    amount: float
    payer_member_id: int = Field(foreign_key="member.id", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExpenseParticipant(SQLModel, table=True):
    expense_id: int = Field(foreign_key="expense.id", primary_key=True)
    member_id: int = Field(foreign_key="member.id", primary_key=True)


# -----------------------------
# App + DB init
# -----------------------------
app = FastAPI(title="Vacation Splitter API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500", "https://delightful-cranachan-75e6bd.netlify.app"],   # tighten later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./vacation_splitter.db")

# Render Postgres URLs sometimes start with postgres://, but SQLAlchemy prefers postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, echo=False)



@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)


# -----------------------------
# Helpers
# -----------------------------
def make_code() -> str:
    # 6-char readable code
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(6))


def compute_settlement(member_ids: List[int], expenses: List[Tuple[int, float, List[int]]]):
    # returns net balances + transfers
    paid: Dict[int, float] = {m: 0.0 for m in member_ids}
    owed: Dict[int, float] = {m: 0.0 for m in member_ids}

    for payer_id, amount, parts in expenses:
        if payer_id not in paid:
            continue
        parts = [p for p in parts if p in owed]
        if not parts:
            continue
        paid[payer_id] += amount
        share = amount / len(parts)
        for p in parts:
            owed[p] += share

    net = {m: round(paid[m] - owed[m], 2) for m in member_ids}

    creditors = [(m, net[m]) for m in member_ids if net[m] > 1e-9]
    debtors = [(m, -net[m]) for m in member_ids if net[m] < -1e-9]  # store positive owed
    creditors.sort(key=lambda t: t[1], reverse=True)
    debtors.sort(key=lambda t: t[1], reverse=True)

    transfers = []
    i = j = 0
    while i < len(debtors) and j < len(creditors):
        d_id, d_amt = debtors[i]
        c_id, c_amt = creditors[j]
        amt = round(min(d_amt, c_amt), 2)
        if amt > 0:
            transfers.append((d_id, c_id, amt))
        d_amt = round(d_amt - amt, 2)
        c_amt = round(c_amt - amt, 2)
        debtors[i] = (d_id, d_amt)
        creditors[j] = (c_id, c_amt)
        if debtors[i][1] <= 1e-9:
            i += 1
        if creditors[j][1] <= 1e-9:
            j += 1

    return {"paid": paid, "owed": owed, "net": net, "transfers": transfers}


# -----------------------------
# API
# -----------------------------
@app.post("/api/trips")
def create_trip(payload: dict):
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Trip name required")

    with Session(engine) as s:
        # ensure unique code
        for _ in range(10):
            code = make_code()
            exists = s.exec(select(Trip).where(Trip.code == code)).first()
            if not exists:
                break
        else:
            raise HTTPException(500, "Could not generate join code")

        trip = Trip(name=name, code=code)
        s.add(trip)
        s.commit()
        s.refresh(trip)
        return {"id": trip.id, "name": trip.name, "code": trip.code, "created_at": trip.created_at.isoformat()}


@app.get("/api/trips/{code}")
def get_trip(code: str):
    code = code.strip().upper()
    with Session(engine) as s:
        trip = s.exec(select(Trip).where(Trip.code == code)).first()
        if not trip:
            raise HTTPException(404, "Trip not found")
        return {"id": trip.id, "name": trip.name, "code": trip.code, "created_at": trip.created_at.isoformat()}


@app.get("/api/trips/{code}/members")
def list_members(code: str):
    code = code.strip().upper()
    with Session(engine) as s:
        trip = s.exec(select(Trip).where(Trip.code == code)).first()
        if not trip:
            raise HTTPException(404, "Trip not found")
        members = s.exec(select(Member).where(Member.trip_id == trip.id).order_by(Member.created_at)).all()
        return [{"id": m.id, "name": m.name} for m in members]


@app.post("/api/trips/{code}/members")
def add_member(code: str, payload: dict):
    code = code.strip().upper()
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Name required")

    with Session(engine) as s:
        trip = s.exec(select(Trip).where(Trip.code == code)).first()
        if not trip:
            raise HTTPException(404, "Trip not found")

        # prevent exact duplicates
        existing = s.exec(select(Member).where(Member.trip_id == trip.id, Member.name == name)).first()
        if existing:
            return {"id": existing.id, "name": existing.name}

        m = Member(trip_id=trip.id, name=name)
        s.add(m)
        s.commit()
        s.refresh(m)
        return {"id": m.id, "name": m.name}


@app.get("/api/trips/{code}/expenses")
def list_expenses(code: str):
    code = code.strip().upper()
    with Session(engine) as s:
        trip = s.exec(select(Trip).where(Trip.code == code)).first()
        if not trip:
            raise HTTPException(404, "Trip not found")

        expenses = s.exec(select(Expense).where(Expense.trip_id == trip.id).order_by(Expense.created_at.desc())).all()
        if not expenses:
            return []

        exp_ids = [e.id for e in expenses if e.id is not None]
        parts = s.exec(select(ExpenseParticipant).where(ExpenseParticipant.expense_id.in_(exp_ids))).all()

        parts_map: Dict[int, List[int]] = {}
        for p in parts:
            parts_map.setdefault(p.expense_id, []).append(p.member_id)

        return [
            {
                "id": e.id,
                "description": e.description,
                "amount": e.amount,
                "payer_member_id": e.payer_member_id,
                "participants": parts_map.get(e.id, []),
                "created_at": e.created_at.isoformat(),
            }
            for e in expenses
        ]


@app.post("/api/trips/{code}/expenses")
def add_expense(code: str, payload: dict):
    code = code.strip().upper()
    desc = (payload.get("description") or "").strip()
    payer_id = payload.get("payer_member_id")
    participants = payload.get("participants") or []
    amount = payload.get("amount")

    if not desc:
        raise HTTPException(400, "Description required")
    try:
        amount = float(amount)
    except Exception:
        raise HTTPException(400, "Amount must be a number")
    if not (amount > 0):
        raise HTTPException(400, "Amount must be > 0")
    if not isinstance(payer_id, int):
        raise HTTPException(400, "payer_member_id must be an int")
    if not isinstance(participants, list) or not all(isinstance(x, int) for x in participants):
        raise HTTPException(400, "participants must be a list of ints")
    if len(participants) == 0:
        raise HTTPException(400, "Select at least one participant")

    with Session(engine) as s:
        trip = s.exec(select(Trip).where(Trip.code == code)).first()
        if not trip:
            raise HTTPException(404, "Trip not found")

        # basic trip membership validation
        member_ids = set(s.exec(select(Member.id).where(Member.trip_id == trip.id)).all())
        if payer_id not in member_ids:
            raise HTTPException(400, "Payer is not a member of this trip")
        if any(p not in member_ids for p in participants):
            raise HTTPException(400, "One or more participants are not members of this trip")

        e = Expense(trip_id=trip.id, description=desc, amount=amount, payer_member_id=payer_id)
        s.add(e)
        s.commit()
        s.refresh(e)

        rows = [ExpenseParticipant(expense_id=e.id, member_id=pid) for pid in participants]
        for r in rows:
            s.add(r)
        s.commit()

        return {"id": e.id}


@app.get("/api/trips/{code}/settlement")
def settlement(code: str):
    code = code.strip().upper()
    with Session(engine) as s:
        trip = s.exec(select(Trip).where(Trip.code == code)).first()
        if not trip:
            raise HTTPException(404, "Trip not found")

        members = s.exec(select(Member).where(Member.trip_id == trip.id)).all()
        member_ids = [m.id for m in members if m.id is not None]
        names = {m.id: m.name for m in members}

        expenses = s.exec(select(Expense).where(Expense.trip_id == trip.id)).all()
        if not expenses:
            return {"members": names, "net": {}, "transfers": []}

        exp_ids = [e.id for e in expenses if e.id is not None]
        parts = s.exec(select(ExpenseParticipant).where(ExpenseParticipant.expense_id.in_(exp_ids))).all()

        parts_map: Dict[int, List[int]] = {}
        for p in parts:
            parts_map.setdefault(p.expense_id, []).append(p.member_id)

        packed = [(e.payer_member_id, float(e.amount), parts_map.get(e.id, [])) for e in expenses]
        res = compute_settlement(member_ids, packed)

        # Convert to readable output
        net_named = {names[mid]: res["net"][mid] for mid in member_ids}
        transfers_named = [(names[a], names[b], amt) for (a, b, amt) in res["transfers"]]

        return {"members": names, "net": net_named, "transfers": transfers_named}

