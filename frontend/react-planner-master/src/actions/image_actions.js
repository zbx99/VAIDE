import {SET_SD_IMAGE, SET_UPLOAD_IMAGE} from "../constants";

export function setUploadImage(uploadImage) {
  return {
    type: SET_UPLOAD_IMAGE,
    uploadImage
  };
}

export function setSDImage(sdImage) {
  return {
    type: SET_SD_IMAGE,
    sdImage
  };
}
