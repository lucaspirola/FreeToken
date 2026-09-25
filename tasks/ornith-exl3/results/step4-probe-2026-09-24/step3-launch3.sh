#!/bin/bash
mv /root/s3-status /root/s3-status-attempt1 2>/dev/null
for r in 0.90 0.85; do
  RATIO=$r flock /root/gpu.lock /root/step3.sh
  grep -q "probe rc=0" /root/s3-status && break
  echo "ratio $r failed, falling back" >> /root/s3-status
done
echo ALLDONE >> /root/s3-status
