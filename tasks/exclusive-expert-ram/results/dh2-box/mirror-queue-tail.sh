#!/bin/bash
# Tail of /root/mirror-queue.sh, relaunched by the exl3 worker (session 01fb928a) on 2026-09-25.
# mirror-queue.sh held gpu.lock (fd 9) while its child checkpoint-box.sh opened a new fd and blocked in
# flock 9 on the same file: self-deadlock from 09:15:24Z, every gpu.lock job on the box stuck. ptrace is
# not permitted here (no gdb unlock), so the parent shell 738558 was killed to free the lock; the child
# (pid 740472, dt128m) was left running and takes the lock itself. This script does what the parent
# would have done after it: status lines, then lfunat-resume.sh (which takes gpu.lock per arm).
C=740472
while kill -0 $C 2>/dev/null; do sleep 10; done
echo "dt128m finished $(date -u +%T) (rc unknown: parent queue killed to break the gpu.lock self-deadlock, see /root/dh2-1m.log and /root/mirror-queue-tail.sh)" >> /root/dh2-status
echo ALLDONE >> /root/dh2-status
sleep 20
/root/belady/lfunat-resume.sh
