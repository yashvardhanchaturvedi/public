from flask import Flask, render_template, Response,request
from video_final import VideoCamera

#Default coordinates of square and circle respectively
startX=100
endX=300
startY=50
endY=250
ptX=200
ptY=150
radius=100

shape="square" #Default boundary shape is square

app = Flask(__name__)

text="" # For person detection text i.e Authorized person or not
qr="" #For QR code text 
start="no" #Flag to check if person detection button is clicked or not
detect="no" #FLag to check if facial detection button is clicked
license="no" #Flag for license plate detection
hem="no"  #Flag for helmet detection
#Main route for the webpage
@app.route('/index',methods=["POST","GET"])
def index():
	if(request.method=="POST"):
		global text,startX,startY,endX,endY,shape,ptX,ptY,shape,radius,detect,hem,start,license
		content=request.data
		content=request.json
		if "start" in content:
			start=content["start"]
		
		if "detect" in content:
			detect=content["detect"]

		if "license" in content:
			license=content["license"]
		
		if "helmet_detect" in content:
			hem=content["helmet_detect"]

		if "type1" in content:
			shape=content["type1"]
			return render_template("index.html")
		else:
			if "startX" in content:
				startX=int(content["startX"])
				startY=int(content["startY"])
				endX=int(content["endX"])
				endY=int(content["endY"])
				return render_template("index.html")
			if "ptX" in content:
				ptX=int(content["ptX"])
				ptY=int(content["ptY"])
				radius=int(content["radius"])
				return render_template("index.html")
		return render_template("index.html")
	elif(request.method=="GET"):
		return render_template("index.html")


#Route for displaying message of the safety intrusion
@app.route('/display',methods=["GET"])
def message():
	global text 
	return text

#Route for qrcode button
@app.route('/displayqr',methods=["GET"])
def messageqr():
	global qr
	return qr
i=0

def gen(camera):
	global startX,startY,endX,endY,text,qr,start,shape,i,license,hem
	print(i)
	#infinite loop for continuos capture of frames
	while True:
		global detect
		if(shape=="square"):
			frame,text,qr,det = VideoCamera.get_frame(VideoCamera,"square",bound_lowX=startX,bound_highX=endX,bound_lowY=startY,bound_highY=endY,detect=detect,start=start,license=license,helmet_detect=hem)
		else:
			frame,text,qr,detect = VideoCamera.get_frame(camera,"circle",ptX=ptX,ptY=ptY,radius=radius,detect=detect,start=start,license=license,helmet_detect=hem)
		yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



#Route for dispaling live stream on the webpage
@app.route('/video_feed1')
def video_feed1():
			global detect
			return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)