import serial# import Serial
from autopy import mouse
#from autopy.mouse import LEFT_BUTTON, RIGHT_BUTTON
ser = serial.Serial('COM3',9600)
mouse.move(683,359)
x=0
y=0
while 1:
    try:
        x,y = mouse.location()
        acc = map(float,ser.readline().split(','))
        print(str(x+acc[0]) +"    " +str(y+acc[1]))
        t=x+acc[0]
        e=y-acc[1]
        if t>=1366:
            t=1360
        elif t<=0:
            t=5
        if e>=766:
            e=760
        elif e<=0:
            e=5
        mouse.move(t,e)
        if acc[3]==1:
            mouse.click(RIGHT_BUTTON)

           
      
        if acc[4]==1:
            mouse.click() 
        
        
    except:
        continue

    #pp[0] = (ser.readline().split())
    #print pp
        #print(accelerometerReading)
    #x=accelerometerReading.rfind
    #print(x)
    #t=acceloremeterReading[x+1:]
    #y=int(t)
    #print(accelerometerReading)