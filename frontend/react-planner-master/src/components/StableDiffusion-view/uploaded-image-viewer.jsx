import React from 'react';
import PropTypes from 'prop-types';
import {Panel, Image} from 'react-bootstrap';
import {browserImageUpload, sendPostRequest} from "../../utils/browser";
import { State } from '../../models';
import {stableDiffusionImageActions} from "../../actions/export";

export default function UploadedImageViewer({state,stableDiffusionImageActions}) {
  let loadImages = event => {
    event.preventDefault();
    browserImageUpload().then((data) => {
      stableDiffusionImageActions.setUploadImage(data)
      // sendPostRequest(data).then((json)=>{
      //   let sdImageResult = JSON.parse(json)
      //   let sdImage = `data:image/png;base64,`+sdImageResult.images[0]
      //   stableDiffusionImageActions.setSDImage(sdImage)
      // });
    });
  };
  return (
      <Panel style={{ height: '100%' }}>
        <Panel.Heading>
          <Panel.Title componentClass="h4">Upload Image</Panel.Title>
        </Panel.Heading>
        <Panel.Body onClick={loadImages}>
          <Image src={state.uploadImage} rounded style={{ maxWidth: '100%', maxHeight: '100%'}} />
        </Panel.Body>
      </Panel>
    )
}
