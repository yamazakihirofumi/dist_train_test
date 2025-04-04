import socket
import subprocess
import re

def get_network_interface(specified_interface=None):
    """Get network interface with private IP for distributed training."""
    if specified_interface:
        return specified_interface
    
    try:
        # Try to find interface with private IP using socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 1))  # Doesn't actually connect
        ip = s.getsockname()[0]
        s.close()
        
        # Find which interface has this IP
        output = subprocess.check_output(['ip', 'addr'], text=True, stderr=subprocess.DEVNULL)
        pattern = fr'\d+: (\w+).+?inet {ip}/'
        match = re.search(pattern, output, re.DOTALL)
        if match:
            return match.group(1)
        
        # If we got an IP but couldn't match interface, try finding any private IP
        for line in output.split('\n'):
            if 'inet ' in line:
                match = re.search(r'inet (192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2[0-9]|3[0-1])\.\d+\.\d+).+?(\w+)$', line)
                if match:
                    return match.group(3)
    except:
        pass
    
    # Fallback to common interface names
    for iface in ['eth0', 'ens33', 'wlan0', 'wlp4s0', 'en0']:
        try:
            subprocess.check_output(['ip', 'link', 'show', iface], stderr=subprocess.DEVNULL)
            return iface
        except:
            continue
    
    return "lo"  # Last resort



def get_ip_address(interface=None):
    if interface:
        try:
            # Try to get IP of specific interface
            output = subprocess.check_output(['ip', 'addr', 'show', interface], text=True, stderr=subprocess.DEVNULL)
            match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', output)
            if match:
                return match.group(1)
        except (subprocess.SubprocessError, FileNotFoundError):
            # If 'ip' command not available (e.g., on macOS)
            pass
            
    # Try to find a good IP using socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 1))  # Doesn't actually connect
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        pass
    
    # Try to find any private IP
    try:
        output = subprocess.check_output(['ip', 'addr'], text=True, stderr=subprocess.DEVNULL)
        for line in output.split('\n'):
            if 'inet ' in line:
                # Check for private IP ranges
                match = re.search(r'inet (192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2[0-9]|3[0-1])\.\d+\.\d+)', line)
                if match:
                    return match.group(1)
    except:
        pass
        
    # Last resort
    return '127.0.0.1'