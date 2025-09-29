TODO
----
- Convert scripts for demod/mod (done)
- Add command line arg for URH (done)
- --urh-demod (done)
- --urh-mod (done)
- Capture raw demods for speed=0 at all addresses
- Feed to chatgpt to update affine transformation

Specs
-----
433.9M
-+ 31kHz FSK
1M sample rate
500 samples/symbol

RECV cmd line
-------------
urh_cli -d USRP -f 433.9e6 -b 200e3 -s 200e3 -mo FSK --gain 20 -sps 100 -pm -31000 31000 -cf 0 -p 0 -rx -rt -1 -a | ./fan433.py decode

TRANSMIT
--------
./fan433.py genwave --speed 0 --addr 7 | xargs urh_cli -d USRP -f 433.9e6 -b 200e3 -s 200e3 -mo FSK --gain 70 -sps 100 -tx -v -pm -31000 31000 -cf 0 -p 0 -m
