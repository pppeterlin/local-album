#!/usr/bin/env python3
"""
manage_users.py — admin CLI for users/groups of the face-naming UI.

Examples:
  # First-time bootstrap: create an admin (you)
  uv run python src/manage_users.py user-add chun --role admin --password 'xxx'

  # Add a viewer with one or more groups
  uv run python src/manage_users.py user-add mom --role viewer --groups family --password 'xxx'
  uv run python src/manage_users.py user-set-password mom --password 'newpw'
  uv run python src/manage_users.py user-set-groups mom --groups family,siblings

  uv run python src/manage_users.py user-list
  uv run python src/manage_users.py user-rm guest

  # Define a visibility group
  uv run python src/manage_users.py group-set family \
      --allowed-faces face_5,face_252,face_0 \
      --blocked-paths /Volumes/.../8_Xiaomi_Mi13Ultra/,/Volumes/.../9_Xiaomi_Mi17/

  uv run python src/manage_users.py group-list
  uv run python src/manage_users.py group-rm family
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _auth import (  # noqa: E402
    AUTH_DIR, USERS_FILE, GROUPS_FILE,
    hash_password, load_users, save_users, load_groups, save_groups,
)


def _split_csv(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


# ---- user commands ---------------------------------------------------------

def cmd_user_add(args):
    users = load_users()
    if args.username in users:
        print(f"user '{args.username}' already exists; use user-set-password / user-set-groups", file=sys.stderr)
        return 1
    if args.role not in ("admin", "viewer"):
        print(f"role must be admin or viewer", file=sys.stderr)
        return 1
    users[args.username] = {
        "password_hash": hash_password(args.password),
        "role": args.role,
        "identity": (args.identity or "").strip(),
        "groups": _split_csv(args.groups),
    }
    save_users(users)
    print(f"✓ user '{args.username}' added (role={args.role}, identity={users[args.username]['identity'] or '訪客'}, groups={users[args.username]['groups']})")
    return 0


def cmd_user_set_identity(args):
    users = load_users()
    if args.username not in users:
        print(f"no such user: {args.username}", file=sys.stderr)
        return 1
    users[args.username]["identity"] = (args.identity or "").strip()
    save_users(users)
    print(f"✓ identity for '{args.username}': {users[args.username]['identity'] or '訪客'}")
    return 0


def cmd_user_set_password(args):
    users = load_users()
    if args.username not in users:
        print(f"no such user: {args.username}", file=sys.stderr)
        return 1
    users[args.username]["password_hash"] = hash_password(args.password)
    save_users(users)
    print(f"✓ password updated for '{args.username}'")
    return 0


def cmd_user_set_groups(args):
    users = load_users()
    if args.username not in users:
        print(f"no such user: {args.username}", file=sys.stderr)
        return 1
    users[args.username]["groups"] = _split_csv(args.groups)
    save_users(users)
    print(f"✓ groups for '{args.username}': {users[args.username]['groups']}")
    return 0


def cmd_user_list(_):
    users = load_users()
    if not users:
        print("(no users; use user-add to create one)")
        return 0
    for name, u in sorted(users.items()):
        ident = u.get("identity") or "訪客"
        print(f"  {name:15} role={u.get('role','viewer'):7} identity={ident:18} groups={u.get('groups', [])}")
    return 0


def cmd_user_rm(args):
    users = load_users()
    if args.username not in users:
        print(f"no such user: {args.username}", file=sys.stderr)
        return 1
    del users[args.username]
    save_users(users)
    print(f"✓ user '{args.username}' removed")
    return 0


# ---- group commands --------------------------------------------------------

def cmd_group_set(args):
    groups = load_groups()
    groups[args.name] = {
        "allowed_faces": _split_csv(args.allowed_faces),
        "blocked_paths": _split_csv(args.blocked_paths),
    }
    save_groups(groups)
    g = groups[args.name]
    print(f"✓ group '{args.name}': {len(g['allowed_faces'])} faces, {len(g['blocked_paths'])} blocked paths")
    return 0


def cmd_group_list(_):
    groups = load_groups()
    if not groups:
        print("(no groups; use group-set to create one)")
        return 0
    for name, g in sorted(groups.items()):
        print(f"  {name}:")
        print(f"    allowed_faces ({len(g.get('allowed_faces', []))}): {g.get('allowed_faces', [])}")
        print(f"    blocked_paths ({len(g.get('blocked_paths', []))}): {g.get('blocked_paths', [])}")
    return 0


def cmd_group_rm(args):
    groups = load_groups()
    if args.name not in groups:
        print(f"no such group: {args.name}", file=sys.stderr)
        return 1
    del groups[args.name]
    save_groups(groups)
    print(f"✓ group '{args.name}' removed")
    return 0


# ---- main ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Manage face-naming UI users / visibility groups")
    sub = p.add_subparsers(dest="cmd", required=True)

    ua = sub.add_parser("user-add", help="create a user")
    ua.add_argument("username")
    ua.add_argument("--role", default="viewer", choices=["admin", "viewer"])
    ua.add_argument("--password", required=True)
    ua.add_argument("--identity", default="", help="face_id this user IS (本人); empty = 訪客")
    ua.add_argument("--groups", help="csv of group names")
    ua.set_defaults(func=cmd_user_add)

    usp = sub.add_parser("user-set-password", help="reset password")
    usp.add_argument("username"); usp.add_argument("--password", required=True)
    usp.set_defaults(func=cmd_user_set_password)

    usg = sub.add_parser("user-set-groups", help="replace user's group list")
    usg.add_argument("username"); usg.add_argument("--groups", required=True, help="csv")
    usg.set_defaults(func=cmd_user_set_groups)

    usi = sub.add_parser("user-set-identity", help="set user's identity face_id (empty = 訪客)")
    usi.add_argument("username"); usi.add_argument("--identity", default="", help="face_id, or empty for 訪客")
    usi.set_defaults(func=cmd_user_set_identity)

    sub.add_parser("user-list").set_defaults(func=cmd_user_list)

    ur = sub.add_parser("user-rm", help="delete a user")
    ur.add_argument("username")
    ur.set_defaults(func=cmd_user_rm)

    gs = sub.add_parser("group-set", help="create or overwrite a visibility group")
    gs.add_argument("name")
    gs.add_argument("--allowed-faces", help="csv of face_ids")
    gs.add_argument("--blocked-paths", help="csv of path prefixes")
    gs.set_defaults(func=cmd_group_set)

    sub.add_parser("group-list").set_defaults(func=cmd_group_list)

    gr = sub.add_parser("group-rm", help="delete a group")
    gr.add_argument("name")
    gr.set_defaults(func=cmd_group_rm)

    args = p.parse_args()
    print(f"(auth files under {AUTH_DIR})")
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
