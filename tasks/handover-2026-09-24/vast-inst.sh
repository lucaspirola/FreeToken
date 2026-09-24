#!/bin/bash
K=$(cat ~/.config/vastai/vast_api_key)
curl -s -m 30 "https://console.vast.ai/api/v0/instances/?owner=me" -H "Authorization: Bearer $K" | python3 -c "
import json,sys
for x in json.load(sys.stdin)['instances']: print(x['id'],x.get('actual_status'),x.get('cur_state'),(x.get('status_msg') or '')[:150].replace('\n',' '),'ssh',x.get('ssh_host'),x.get('ssh_port'),'ip',x.get('public_ipaddr'),x.get('image_runtype'),'dph',round(x.get('dph_total',0),3))"
