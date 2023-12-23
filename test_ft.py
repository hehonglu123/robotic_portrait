#!/usr/bin/env python


import sys
sys.path.append('toolbox')
import rpi_ati_net_ft

def main():
    

    try:
        netft=rpi_ati_net_ft.NET_FT('192.168.60.100')
        netft.set_tare_from_ft()
        print(netft.read_ft_http())
        print(netft.try_read_ft_http())
        
        netft.start_streaming()
        
        while(True):
            #print(netft.read_ft_streaming(.1))
            res, ft, status = netft.try_read_ft_streaming(.1)
            # print("res: ", res)
            print("ft: ", ft)
            # print("status: ", status)
        
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()