import blobconverter
import cv2
import depthai as dai
import numpy as np
import RobotRaconteur as RR
RRN=RR.RobotRaconteurNode.s


face_tracking_interface="""
service experimental.face_tracking

import com.robotraconteur.image
import com.robotraconteur.imaging.camerainfo
import com.robotraconteur.param
import com.robotraconteur.device
import com.robotraconteur.device.isoch
import com.robotraconteur.device.clock
import com.robotraconteur.datetime

using com.robotraconteur.image.Image

object face_tracking_obj
	wire single[] bbox [readonly]
	wire Image frame_stream [readonly]
end object
"""





class face_tracking_impl(object):

	def __init__(self) -> None:
		self._image_type = RRN.GetStructureType('com.robotraconteur.image.Image')
	def update(self,bbox):
		self.bbox.OutValue=bbox
	def update_frame(self,mat):
		self.frame_stream.OutValue=self._cv_mat_to_image(mat)
	
	def _cv_mat_to_image(self, mat):

		is_mono = False
		if (len(mat.shape) == 2 or mat.shape[2] == 1):
			is_mono = True

		image_info = self._image_info_type()
		image_info.width =mat.shape[1]
		image_info.height = mat.shape[0]
		if is_mono:
			image_info.step = mat.shape[1]
			image_info.encoding = self._image_consts["ImageEncoding"]["mono8"]
		else:
			image_info.step = mat.shape[1]*3
			image_info.encoding = self._image_consts["ImageEncoding"]["bgr888"]
		image_info.data_header = self._sensor_data_util.FillSensorDataHeader(self._camera_info.device_info,self._seqno)
		

		image = self._image_type()
		image.image_info = image_info
		image.data=mat.reshape(mat.size, order='C')
		return image

class HostSync:
	def __init__(self):
		self.arrays = {}
	def add_msg(self, name, msg):
		if not name in self.arrays:
			self.arrays[name] = []
		self.arrays[name].append(msg)
	def get_msgs(self, seq):
		ret = {}
		for name, arr in self.arrays.items():
			for i, msg in enumerate(arr):
				if msg.getSequenceNum() == seq:
					ret[name] = msg
					self.arrays[name] = arr[i:]
					break
		return ret

def create_pipeline():
	print("Creating pipeline...")
	pipeline = dai.Pipeline()

	# ColorCamera
	print("Creating Color Camera...")
	cam = pipeline.create(dai.node.ColorCamera)
	cam.setPreviewSize(300, 300)
	cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
	cam.setVideoSize(1080,1080)
	cam.setInterleaved(False)

	cam_xout = pipeline.create(dai.node.XLinkOut)
	cam_xout.setStreamName("frame")
	cam.video.link(cam_xout.input)

	# NeuralNetwork
	print("Creating Face Detection Neural Network...")
	face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
	face_det_nn.setConfidenceThreshold(0.5)
	face_det_nn.setBlobPath(blobconverter.from_zoo(
		name="face-detection-retail-0004",
		shaves=6,
	))
	# Link Face ImageManip -> Face detection NN node
	cam.preview.link(face_det_nn.input)

	objectTracker = pipeline.create(dai.node.ObjectTracker)
	objectTracker.setDetectionLabelsToTrack([1])  # track only person
	# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
	objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
	# take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
	objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

	# Linking
	face_det_nn.passthrough.link(objectTracker.inputDetectionFrame)
	face_det_nn.passthrough.link(objectTracker.inputTrackerFrame)
	face_det_nn.out.link(objectTracker.inputDetections)
	# Send face detections to the host (for bounding boxes)

	pass_xout = pipeline.create(dai.node.XLinkOut)
	pass_xout.setStreamName("pass_out")
	objectTracker.passthroughTrackerFrame.link(pass_xout.input)

	tracklets_xout = pipeline.create(dai.node.XLinkOut)
	tracklets_xout.setStreamName("tracklets")
	objectTracker.out.link(tracklets_xout.input)
	print("Pipeline created.")
	return pipeline



with RR.ServerNodeSetup("experimental.face_tracking", 52222):
	#Register the service type
	RRN.RegisterServiceType(face_tracking_interface)

	face_tracking_inst=face_tracking_impl()
	
	#Register the service
	RRN.RegisterService("Face_tracking","experimental.face_tracking.face_tracking_obj",face_tracking_inst)
		
	with dai.Device(create_pipeline()) as device:
		frame_q = device.getOutputQueue("frame")
		tracklets_q = device.getOutputQueue("tracklets")
		pass_q = device.getOutputQueue("pass_out")
		sync=HostSync()
		while True:
			sync.add_msg("color", frame_q.get())

			# Using tracklets instead of ImgDetections in case NN inaccuratelly detected face, so blur
			# will still happen on all tracklets (even LOST ones)
			nn_in = tracklets_q.tryGet()
			if nn_in is not None:
				seq = pass_q.get().getSequenceNum()
				msgs = sync.get_msgs(seq)

				if not 'color' in msgs: continue
				frame = msgs["color"].getCvFrame()

				bboxes=[]
				bboxes_area=[]
				for t in nn_in.tracklets:
					# Expand the bounding box a bit so it fits the face nicely (also convering hair/chin/beard)
					t.roi.x -= t.roi.width / 10
					t.roi.width = t.roi.width * 1.2
					t.roi.y -= t.roi.height / 7
					t.roi.height = t.roi.height * 1.2

					roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
					bbox = [int(roi.topLeft().x), int(roi.topLeft().y), int(roi.bottomRight().x), int(roi.bottomRight().y)]

					face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
					fh, fw, fc = face.shape
					frame_h, frame_w, frame_c = frame.shape

					###if any bbox vertices are out of frame, ignore
					if bbox[0]<0 or bbox[1]<0 or bbox[2]>frame_w or bbox[3]>frame_h:
						continue

					#draw bounding box
					cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
					
					bboxes.append(bbox)
					bboxes_area.append((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
					


				cv2.imshow("Frame", cv2.resize(frame, (900,900)))

				###find the largest bbox
				if len(bboxes)>0:
					bbox=bboxes[np.argmax(bboxes_area)]
				else:
					bbox=[]
				face_tracking_inst.update(bbox)

			if cv2.waitKey(1) == ord('q'):
				break
