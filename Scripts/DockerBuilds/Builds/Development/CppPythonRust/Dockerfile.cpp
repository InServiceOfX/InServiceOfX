RUN apt-get -y install build-essential ccache libboost-all-dev \
  libtbb-dev

# Install prerequisites for Kitware repository
RUN apt-get install -y ca-certificates gpg wget
 
# If kitware-archive-keyring package hasn't been installed previously, manually
# obtain a copy of signing key:
RUN (test -f /usr/share/doc/kitware-archive-keyring/copyright) || \
  (wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
    gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null) && \
  # Add repository to sources list and update.
  # For Ubuntu Jammy Jellyfish (22.04):
  echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] \
    https://apt.kitware.com/ubuntu/ noble main' | \
      tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
  apt-get update && \
  # If kitware-archive-keyring package hasn't been installed previously, remove
  # manually obtained signed key to make room for package:
  test -f /usr/share/doc/kitware-archive-keyring/copyright || \
  rm /usr/share/keyrings/kitware-archive-keyring.gpg && \
  # Install kitware-archive-keyring package to ensure keyring stays up to date
  # as kitware rotates keys.
  apt-get install kitware-archive-keyring && \
  apt-get install -y cmake  

