class StableDiffusionImage {
  static setUploadImage( state, uploadImage ){
    return { updatedState: state.set('uploadImage', uploadImage ) };
  }
  static setSDImage( state, sdImage ){
    return { updatedState: state.set('sdImage', sdImage ) };
  }
}

export { StableDiffusionImage as default };
