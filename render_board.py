import json, pathlib
tmpl = open('template.html').read()
bundle = json.load(open('out/board_data_v3.json'))
data = json.dumps(bundle, separators=(',',':'))
html = tmpl.replace('__DATA__', data)
pathlib.Path('public').mkdir(exist_ok=True)
open('public/index.html','w').write(html)
open('out/td-board.html','w').write(html)
print(f"rendered public/index.html ({len(html)} bytes, {len(bundle['players'])} players)")
