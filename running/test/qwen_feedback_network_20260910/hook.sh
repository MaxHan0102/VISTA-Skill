#!/bin/sh
set -eu
case "$reason" in
 BOUND|RENEW|REBIND|REBOOT)
  case "$new_ip_address" in 192.168.1.*) ;; *) echo "Unexpected DHCP subnet" >&2; exit 1;; esac
  ip address replace "$new_ip_address/32" dev "$interface" label "$interface:qwen"
  for target in 192.168.1.173 192.168.1.185; do
   ip route replace "$target/32" dev "$interface" src "$new_ip_address" proto static metric 50
  done
  printf '%s %s %s\n' "$reason" "$new_ip_address" "${new_dhcp_server_identifier:-unknown}" >> /tmp/qwen-feedback-network/bindings.log
  ;;
 EXPIRE|FAIL|RELEASE|STOP)
  if [ -n "${old_ip_address:-}" ]; then
   for target in 192.168.1.173 192.168.1.185; do ip route del "$target/32" dev "$interface" src "$old_ip_address" metric 50 2>/dev/null || true; done
   ip address del "$old_ip_address/32" dev "$interface" 2>/dev/null || true
  fi
  ;;
esac
exit 0
