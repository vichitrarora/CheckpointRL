"""Publish 16 alternating submission commits using the configured dates.

Uses only stdlib and Git. Each participant runs their own wrapper/account.
Dates apply to child Git processes only; no global configuration is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def git(folder, *args, env=None, check=True):
    result = subprocess.run(
        ["git", "-c", "gc.auto=0", "-C", str(folder), *args],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    if check and result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout.strip() if check else result


def snapshot(source, plan):
    digest = hashlib.sha256()
    seen = set()
    for entry in plan:
        for relative in entry["files"]:
            path = (source / relative).resolve()
            if not path.is_relative_to(source) or relative in seen:
                raise ValueError(f"Unsafe or duplicated path: {relative}")
            seen.add(relative)
            data = path.read_bytes()
            if path.suffix not in {".png", ".pdf", ".zip", ".npz"}:
                data = data.replace(b"\r\n", b"\n")
            digest.update(relative.encode() + b"\0" + hashlib.sha256(data).digest())
    return digest.hexdigest()


def inspect_history(destination, plan, fingerprint, ref="HEAD"):
    if git(destination, "rev-parse", "--verify", ref, check=False).returncode:
        return 0
    commits = git(destination, "rev-list", "--reverse", ref).splitlines()
    if len(commits) > len(plan):
        raise RuntimeError("Destination has commits outside this 16-step plan.")
    for index, commit in enumerate(commits):
        item = plan[index]
        body = git(destination, "show", "-s", "--format=%B", commit)
        expected = [
            f"Submission-Step: {index + 1}/16",
            f"Submission-Owner: {item['owner']}",
            f"Submission-Snapshot: {fingerprint}",
        ]
        dates = git(destination, "show", "-s", "--format=%aI%n%cI", commit).splitlines()
        if body.splitlines()[0] != item["message"] or not all(x in body.splitlines() for x in expected):
            raise RuntimeError(f"Commit {commit} does not match this source/plan. No history will be overwritten.")
        if dates != [item["date"], item["date"]]:
            raise RuntimeError(f"Unexpected dates in commit {commit}.")
        changed = git(destination, "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", commit).splitlines()
        if set(changed) != set(item["files"]):
            raise RuntimeError(f"Unexpected files in commit {commit}.")
    return len(commits)


def publish(args):
    source = args.source.resolve()
    destination = args.destination.resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and destination must be separate folders, with neither inside the other.")
    plan = json.loads((source / "scripts/commit_plan.json").read_text(encoding="utf-8"))
    if len(plan) != 16 or [x["owner"] for x in plan] != ["vichitr", "jaineel"] * 8:
        raise ValueError("Expected 16 alternating commits.")
    fingerprint = snapshot(source, plan)
    if not (destination / ".git").exists():
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("Destination must be empty or a clone created by this script.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", args.remote, str(destination)], check=True)
    if git(destination, "remote", "get-url", "origin") != args.remote:
        raise ValueError("Destination origin differs from the requested remote.")
    if git(destination, "status", "--porcelain"):
        raise ValueError("Destination contains local changes. Review them before retrying.")
    branch = git(destination, "symbolic-ref", "--short", "HEAD")
    if branch != "main":
        if git(destination, "rev-parse", "--verify", "HEAD", check=False).returncode:
            git(destination, "symbolic-ref", "HEAD", "refs/heads/main")
        else:
            raise ValueError("Destination must be on main.")
    git(destination, "config", "user.name", args.name)
    git(destination, "config", "user.email", args.email)
    # Match the exported snapshot on Windows and Unix.
    git(destination, "config", "core.autocrlf", "false")
    deadline = time.monotonic() + args.wait_minutes * 60
    while True:
        if snapshot(source, plan) != fingerprint:
            raise RuntimeError("Source changed during publication. Restore the identical shared snapshot.")
        remote_ref = git(destination, "ls-remote", "origin", "refs/heads/main")
        remote_count = 0
        if remote_ref:
            git(destination, "fetch", "origin", "main")
            remote_count = inspect_history(destination, plan, fingerprint, "FETCH_HEAD")
            if git(destination, "rev-parse", "--verify", "HEAD", check=False).returncode:
                git(destination, "checkout", "-B", "main", "FETCH_HEAD")
            else:
                # Refuses divergence; never reset, force-push, or rewrite another commit.
                git(destination, "merge", "--ff-only", "FETCH_HEAD")
        count = inspect_history(destination, plan, fingerprint)
        if count > remote_count:
            # Resume a commit whose previous push failed, using the same hash/date.
            if count != remote_count + 1 or plan[count - 1]["owner"] != args.owner:
                raise RuntimeError("Unexpected unpublished commits; manual review required.")
            git(destination, "push", "-u", "origin", "main")
            print(f"Resumed push for step {count}/16", flush=True)
            if args.once:
                return
        if count == 16:
            print("Complete: all 16 submission commits are on origin/main.", flush=True)
            return
        item = plan[count]
        if item["owner"] != args.owner:
            if args.once:
                print(f"Waiting for {item['owner']} to publish step {count + 1}/16.", flush=True)
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("Partner wait expired. Rerun the same command to resume safely.")
            print(f"Waiting for {item['owner']}, step {count + 1}/16...", flush=True)
            time.sleep(args.poll_seconds)
            continue
        for relative in item["files"]:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
        git(destination, "add", "--", *item["files"])
        staged = git(destination, "diff", "--cached", "--name-only").splitlines()
        if set(staged) != set(item["files"]):
            raise RuntimeError("Staged paths differ from the plan; inspect the destination.")
        env = os.environ.copy()
        for variable in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
            env.pop(variable, None)
        env["GIT_AUTHOR_DATE"] = item["date"]
        env["GIT_COMMITTER_DATE"] = item["date"]
        body = (
            f"Submission-Step: {count + 1}/16\nSubmission-Owner: {args.owner}\n"
            f"Submission-Snapshot: {fingerprint}"
        )
        git(destination, "commit", "-m", item["message"], "-m", body, env=env)
        git(destination, "push", "-u", "origin", "main")
        print(f"Published {count + 1}/16: {item['message']} ({item['date']})", flush=True)
        deadline = time.monotonic() + args.wait_minutes * 60
        if args.once:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", choices=["vichitr", "jaineel"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--remote", default="https://github.com/vichitrarora/CheckpointRL")
    parser.add_argument("--once", action="store_true", help="Publish at most one turn; useful for manual alternation.")
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--wait-minutes", type=float, default=60)
    args = parser.parse_args()
    if not args.name.strip() or not args.email.strip() or args.poll_seconds <= 0 or args.wait_minutes <= 0:
        parser.error("Identity and positive wait settings are required.")
    try:
        publish(args)
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Stopped: {exc}\n")


if __name__ == "__main__":
    main()
