#!/bin/bash
# mtest_end.sh: stop the test engine and put the fast unit back on 8080.
cd /root/gbench
./srv.sh stop
./deploy_fast.sh
echo "=== $(date +%T) MTEST DONE"
