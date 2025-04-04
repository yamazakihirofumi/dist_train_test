import subprocess
import re

def get_network_interface(specified_interface=None):
    if specified_interface:
        print(f"Using specified network interface: {specified_interface}")
        return specified_interface
    
    # Auto-detect using ifconfig
    try:
        output = subprocess.check_output(['ifconfig'], text=True)
        
        # Simple regex pattern to find interfaces with non-loopback IPs
        pattern = r'(\w+):.+?inet\s+(?:addr:)?(\d+\.\d+\.\d+\.\d+)'
        matches = re.findall(pattern, output, re.DOTALL)
        
        # Filter for non-loopback IPs (not 127.x.x.x)
        valid_interfaces = [(iface, ip) for iface, ip in matches 
                           if not ip.startswith('127.')]
        
        if valid_interfaces:
            selected = valid_interfaces[0][0]
            print(f"Auto-detected interface: {selected} with IP {valid_interfaces[0][1]}")
            return selected
            
        raise ValueError("No suitable network interface found")
    
    except Exception as e:
        print(f"Error detecting network interface: {e}")
        raise ValueError(f"Failed to detect network interface: {e}")

        