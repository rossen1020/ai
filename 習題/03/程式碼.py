import random  

def maximize():
    x = 0                                                            #先初始x,y,z值
    y = 0
    z = 0
    for i in range(10000):                                           #因為隨機測試所以設 10000 次迭代，確保生成足夠的新的隨機解
        
        x1 = random.uniform(0,10)                                    #經爵寬說明設置0~10的範圍是因為x+y<=10, 所以最大值不可能超過10
        y1 = random.uniform(0,10)  
        z1 = random.uniform(0,5.5)                                   #會設置最大值是因為題目y+2z<=11, 所以z的最大值為5.5
      
        if(x1+y1<=10 and 2*x1+z1<=9 and y1+2*z1<=11):                #對於題目開始計算最佳解
            if 3*x1+2*y1+5*z1>=3*x+2*y+5*z:
                x=x1
                y=y1
                z=z1
                print(f"x: {x:.3f}, y: {y:.3f}, z: {z:.3f}")          #縮減小數位比較好觀察
    ans = 3*x + 2*y + 5*z                                             #針對最佳結果進行題目要求的最大化計算
    return {"x": round(x, 3), "y": round(y, 3), "z": round(z, 3)}, round(ans, 3)

solution, ans = maximize()
print(solution, ans)
