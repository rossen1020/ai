import random
courses = [
  {'teacher': '  ', 'name':'　　', 'hours': -1}, 
  {'teacher': '甲', 'name':'機率', 'hours': 2},
  {'teacher': '甲', 'name':'線代', 'hours': 3},
  {'teacher': '甲', 'name':'離散', 'hours': 3},
  {'teacher': '乙', 'name':'視窗', 'hours': 3},
  {'teacher': '乙', 'name':'科學', 'hours': 3},
  {'teacher': '乙', 'name':'系統', 'hours': 3},
  {'teacher': '乙', 'name':'計概', 'hours': 3},
  {'teacher': '丙', 'name':'軟工', 'hours': 3},
  {'teacher': '丙', 'name':'行動', 'hours': 3},
  {'teacher': '丙', 'name':'網路', 'hours': 3},
  {'teacher': '丁', 'name':'媒體', 'hours': 3},
  {'teacher': '丁', 'name':'工數', 'hours': 3},
  {'teacher': '丁', 'name':'動畫', 'hours': 3},
  {'teacher': '丁', 'name':'電子', 'hours': 4},
  {'teacher': '丁', 'name':'嵌入', 'hours': 3},
  {'teacher': '戊', 'name':'網站', 'hours': 3},
  {'teacher': '戊', 'name':'網頁', 'hours': 3},
  {'teacher': '戊', 'name':'演算', 'hours': 3},
  {'teacher': '戊', 'name':'結構', 'hours': 3},
  {'teacher': '戊', 'name':'智慧', 'hours': 3}
]

teachers = ['甲', '乙', '丙', '丁', '戊']

rooms = ['A', 'B']

slots = [
  'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17',
  'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27',
  'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37',
  'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47',
  'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57',
  'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17',
  'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27',
  'B31', 'B32', 'B33', 'B34', 'B35', 'B36', 'B37',
  'B41', 'B42', 'B43', 'B44', 'B45', 'B46', 'B47',
  'B51', 'B52', 'B53', 'B54', 'B55', 'B56', 'B57',
]


def randomclass():
    cl={}
    for course in courses:
        if course['hours']>0:
            cname=course['name']
            teacher=course['teacher']
            slot=random.choice(slots)
            room=random.choice(rooms)
            cl[cname]={'teacher':teacher, 'slot':slot, 'room':room}
    return cl



def valid(cl):
    for slot in slots:
        teachers_s=set()
        room_s=set()
        for cname, info in cl.items():
            if info['slot']==slot:
                teacher=info['teacher']
                room=info['room']
                if teacher in teachers_s or room in room_s:
                    return False
                teachers_s.add(teacher)
                room_s.add(room)
    return True



def valid_cl():
    while True:
        cl=randomclass()
        if valid(cl):
            return cl


valid_class=valid_cl()
for cname, info in valid_class.items():
    print(f"name:{cname}, teacher:{info['teacher']}, hours:{info['slot']}, room:{info['room']}")
