"""Minimal in-memory auth utilities (not production ready).

Provides:
  - password hashing (bcrypt via passlib)
  - JWT creation & verification (HS256)
  - Dependency to extract current user
  - Simple in-memory user registry

This is intentionally simple for a prototype. In production you would:
  * Use a persistent database
  * Add email verification / password reset flows
  * Rotate secrets & manage via env vars / secret manager
  * Use proper user id (UUID) and additional profile fields
"""
from __future__ import annotations
import os, time
from typing import Optional, Dict
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import jwt, JWTError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JWT_SECRET = os.environ.get("BL_JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS = 60 * 60 * 12  # 12h

pwd_ctx = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# ---------------------------------------------------------------------------
# In-memory user store (username -> record)
# ---------------------------------------------------------------------------
class UserRecord(BaseModel):
    username: str
    email: str
    password_hash: str
    plan: str = "Guest"
    created_at: int

_USERS: Dict[str, UserRecord] = {}

# seed guest demo user (password: guest)
if "guest" not in _USERS:
    _USERS["guest"] = UserRecord(
        username="guest",
        email="guest@example.com",
        password_hash=pwd_ctx.hash("guest"),
        plan="Guest",
        created_at=int(time.time()),
    )

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class PublicUser(BaseModel):
    username: str
    email: str
    plan: str
    created_at: int

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hash_password(pw: str) -> str:
    return pwd_ctx.hash(pw)

def verify_password(pw: str, hashed: str) -> bool:
    try:
        return pwd_ctx.verify(pw, hashed)
    except Exception:
        return False

def create_access_token(username: str) -> str:
    now = int(time.time())
    payload = {"sub": username, "iat": now, "exp": now + ACCESS_TOKEN_EXPIRE_SECONDS}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

# ---------------------------------------------------------------------------
# FastAPI deps
# ---------------------------------------------------------------------------

def get_current_user(token: str = Depends(oauth2_scheme)) -> UserRecord:
    cred_exc = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        username: str | None = payload.get("sub")
        if not username:
            raise cred_exc
    except JWTError:
        raise cred_exc
    user = _USERS.get(username)
    if not user:
        raise cred_exc
    return user

# ---------------------------------------------------------------------------
# Auth endpoint handlers
# ---------------------------------------------------------------------------

def handle_login(form: OAuth2PasswordRequestForm = Depends()):
    user = _USERS.get(form.username)
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    token = create_access_token(user.username)
    return Token(access_token=token)

class SignupRequest(BaseModel):
    username: str
    email: str
    password: str

class SignupResponse(BaseModel):
    user: PublicUser
    token: str


def handle_signup(req: SignupRequest):
    if req.username in _USERS:
        raise HTTPException(status_code=400, detail="Username already exists")
    rec = UserRecord(
        username=req.username,
        email=req.email,
        password_hash=hash_password(req.password),
        plan="Guest",
        created_at=int(time.time()),
    )
    _USERS[req.username] = rec
    token = create_access_token(req.username)
    return SignupResponse(user=PublicUser(**rec.dict()), token=token)

# Public projection

def to_public(user: UserRecord) -> PublicUser:
    return PublicUser(username=user.username, email=user.email, plan=user.plan, created_at=user.created_at)
