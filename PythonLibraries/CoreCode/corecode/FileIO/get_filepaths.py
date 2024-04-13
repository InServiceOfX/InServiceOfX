from pathlib import Path

def get_filepaths(directory, recursive=True, suffix=None, prefix=None):
	"""
	@param suffix - e.g. suffix=".pdf". You need the '.' in ".pdf" because of
	how pathlib Path parses out a suffix.
	"""
	if not directory.is_dir():
		directory = directory.parent

	def _add_checked_filepath(paths, path, suffix, prefix):
		if path.is_file() and (suffix is None or path.suffix == suffix) and \
			(prefix is None or path.name.startswith(prefix)):
			paths.append(path.resolve())

	paths = []

	if recursive:
		# https://docs.python.org/3/library/pathlib.html
		# Path.rglob(pattern, *, case_sensitive=None)
		for path in directory.rglob('*'):
			_add_checked_filepath(paths, path, suffix, prefix)

	else:
		for path in directory.glob('*'):
			_add_checked_filepath(paths, path, suffix, prefix)

	return paths