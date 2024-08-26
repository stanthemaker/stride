# stride
## Installation
https://stridecodes.readthedocs.io/en/latest/download.html
## Contents 
- <font color="#1936C9">fwi.py</font>: main code to run forward and reverse simulation. For example, -nw 4 -nth 6 => uses 24 CPUs
```bash
    mrun -nw '# of worker' -nth '# of threads' python fwi.py 'config .json file path'
``` 
- <font color="#1936C9">readfoler.py</font>: read and convert all .h5 files in the output foler to .png
```bash
    python3 readfoler.py --folder 'FOLDER PATH'
``` 
    