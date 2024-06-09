import {StableDiffusionImage} from '../class/export';
import {SET_SD_IMAGE, SET_UPLOAD_IMAGE} from "../constants";

export default function (state, action){
  switch (action.type) {
    case SET_UPLOAD_IMAGE:
      return StableDiffusionImage.setUploadImage(state, action.uploadImage).updatedState;
    case SET_SD_IMAGE:
      return StableDiffusionImage.setSDImage(state,action.sdImage).updatedState;
    default:
      return state
  }
}
