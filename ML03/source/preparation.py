import jsonpickle
import thulac  # pip 安装版本需要把 thulac/character/CBTaggingDecoder.py 第 170 行中 time.clock() 改成 time.time()
import Mail

with open("trec06c-utf8/label/index", mode='rt') as f:
    input_index = f.read()

indexes = input_index.split('\n')
lac = thulac.thulac()

mail_data = []
for index in indexes[:-1]:  # 最后有一个空行
    label, path = index.split(" ")
    address = path[7:]  # 去掉 ../data
    with open("trec06c-utf8/data" + address, mode='rt') as f:
        uncut = f.read()
    parts = uncut.split('\n\n')
    head = parts[0]
    trunk = '\n\n'.join(parts[1:])
    lines = head.split('\n')
    entry = ""
    header = {}
    for line in lines:  # 拆文件头
        if line.startswith("\t") or line.startswith('    '):  # 继续上一条 entry
            header[entry] += "\n" + line.lstrip()
        else:  # 新 entry
            pars = line.split(": ")  # 防止时间遭拆分
            entry = pars[0]
            header[entry] = ': '.join(pars[1:])
    this_mail = Mail(label, address, header, trunk, lac)
    mail_data.append(this_mail)

frozen = jsonpickle.encode(mail_data)
# frozen = frozen.replace('"py/object": "__main__.Mail"', '"py/object": "Mail.Mail"')
f = open("splited_for_script.json", mode='wt')
f.write(frozen)
f.close()
