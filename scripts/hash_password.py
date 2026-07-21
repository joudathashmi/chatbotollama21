#!/usr/bin/env python3
"""Generate a bcrypt password hash for provisioning a MISA_AUTH_USERS account.

Usage:
    python scripts/hash_password.py 'the-password'
    python scripts/hash_password.py            # prompts (input hidden)

Paste the printed hash into the MISA_AUTH_USERS JSON in your .env, e.g.:

    MISA_AUTH_USERS='[{"username":"analyst1","password_hash":"$2b$12$..."}]'

The hash — not the plaintext — is what the server stores and compares
against at login (app/auth.py:authenticate_user).
"""

from __future__ import annotations

import getpass
import sys

import bcrypt


def main() -> int:
    if len(sys.argv) > 2:
        print("Usage: python scripts/hash_password.py [password]", file=sys.stderr)
        return 2
    password = sys.argv[1] if len(sys.argv) == 2 else getpass.getpass("Password: ")
    if not password:
        print("Empty password — nothing to hash.", file=sys.stderr)
        return 1
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    print(hashed.decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
