"""
Namelist parser for Fortran-style namelist files
"""
import re
import os


class Namelist:
    """Parse and store Fortran-style namelist data"""

    def __init__(self, namelist_file):
        """Initialize and parse namelist file"""
        self.data = {}
        self.namelist_file = namelist_file
        self._parse()

    def _parse(self):
        """Parse namelist file"""
        if not os.path.exists(self.namelist_file):
            raise FileNotFoundError(f"Namelist file not found: {self.namelist_file}")

        with open(self.namelist_file, 'r') as f:
            lines = f.readlines()

        # Remove comments and join lines
        clean_lines = []
        for line in lines:
            # Remove inline comments
            if '!' in line:
                line = line[:line.index('!')]
            line = line.strip()
            if line:
                clean_lines.append(line)

        content = '\n'.join(clean_lines)

        # Find all namelists (between &NAME and /)
        # Use a pattern that matches / at line start or standalone
        pattern = r'&(\w+)(.*?)(?:^/|\n/)'
        matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)

        for name, body in matches:
            self.data[name] = {}

            # Parse key-value pairs
            # Handle both key=value and key = value
            # Value can be quoted string or unquoted
            lines = body.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line or '=' not in line:
                    continue

                # Split on first =
                parts = line.split('=', 1)
                if len(parts) != 2:
                    continue

                key = parts[0].strip()
                value = parts[1].strip()

                # Parse value
                self.data[name][key] = self._parse_value(value)

    def _parse_value(self, value):
        """Parse value and convert to appropriate Python type"""
        value = value.strip()

        # String (single or double quotes)
        if (value.startswith("'") and value.endswith("'")) or \
           (value.startswith('"') and value.endswith('"')):
            return value[1:-1]

        # Boolean
        if value.lower() in ['.true.', 't']:
            return True
        if value.lower() in ['.false.', 'f']:
            return False

        # Try integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def get(self, section, key, default=None):
        """Get value from namelist"""
        try:
            return self.data[section][key]
        except KeyError:
            return default

    def get_section(self, section):
        """Get entire section as dictionary"""
        return self.data.get(section, {})

    def __repr__(self):
        """String representation"""
        lines = ["Namelist contents:"]
        for section, params in self.data.items():
            lines.append(f"\n&{section}")
            for key, value in params.items():
                lines.append(f"  {key} = {value}")
            lines.append("/")
        return '\n'.join(lines)
