def get_maximum_sized_face(face_info):
	"""
	@param face_info-Typically the output of app.get(..) where app is insight's
	FaceAnalysis.
	"""
	return sorted(
		face_info,
		key=lambda x:(
			# only use the maximum face.
			x['bbox'][2] - x['bbox'][0]) * x['bbox'][3] - x['bbox'][1])[-1]