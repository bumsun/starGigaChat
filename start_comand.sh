#!/bin/bash
cd /root/images/ && nohup python3 -u -m http.server 3000 >> /root/images/main.log 2>&1 & echo $! > roman.pid