import os


def get_drive_path():

  home_dir = os.path.expanduser("~")
  valid_paths = [
                 os.path.join(home_dir, "Google Drive"),
                 os.path.join(home_dir, "GoogleDrive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "Google Drive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "GoogleDrive"),
                 os.path.join("C:", os.sep, "GoogleDrive"),
                 os.path.join("C:", os.sep, "Google Drive"),
                 os.path.join("D:", os.sep, "GoogleDrive"),
                 os.path.join("D:", os.sep, "Google Drive"),
                 ]

  drive_path = None
  for path in valid_paths:
    if os.path.isdir(path):
      drive_path = path
      break


  return drive_path