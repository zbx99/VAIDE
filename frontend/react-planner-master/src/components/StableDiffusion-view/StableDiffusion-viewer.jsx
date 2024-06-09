import React, {Component} from 'react';
import PropTypes from 'prop-types';
import TextPromptViewer from "./text-prompt-viewer";
import UploadedImageViewer from "./uploaded-image-viewer";
import {Grid, Row, Col, Panel, Image, Button} from 'react-bootstrap';
import {browserImageUpload, sendRecognizationPostRequest, sendSDPostRequest} from "../../utils/browser";
import ToolbarLoadButton from "../toolbar/toolbar-load-button";

export default function StableDiffusionViewer({state, width, height, translator,stableDiffusionImageActions}){

  let regenerate = event =>{
    sendSDPostRequest(state.uploadImage).then((json)=>{
        let sdImageResult = JSON.parse(json)
        let sdImage = `data:image/png;base64,`+sdImageResult.images[0]
        stableDiffusionImageActions.setSDImage(sdImage);
        sendRecognizationPostRequest(sdImageResult.images[0]);
      });
  }

  return (
    <Panel bsStyle="primary" style = {{width: `${width}px`}}>
      <Panel.Heading style = {{backgroundColor: '#005FAF'}}>
          <Panel.Title componentClass="h3">Stable Diffusion</Panel.Title>
      </Panel.Heading>
      <Panel.Body>
        <Grid style = {{width: '100%'}}>
          <Row style={{ display: 'flex', alignItems: 'stretch' }}>
            <Col lg={6} md={6} sm={6}>
              <UploadedImageViewer state={state} stableDiffusionImageActions={stableDiffusionImageActions} style={{ height: 'auto' }}/>
            </Col>
            <Col lg={6} md={6} sm={6}>
              <TextPromptViewer state={state} style={{ height: 'auto' }}/>
            </Col>
          </Row>
          <Row style={{margin:"1%"}}>
            <Col lg={12} md={12} mdOffset={0} className="text-center" >
              <Button bsSize="large" bsStyle="primary" style = {{backgroundColor: '#005FAF'}} onClick={regenerate} >{translator.t("Generate")}</Button>
            </Col>
          </Row>
        </Grid>
      </Panel.Body>
    </Panel>
  )
}
