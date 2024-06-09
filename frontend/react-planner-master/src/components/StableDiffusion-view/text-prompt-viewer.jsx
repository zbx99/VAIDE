import React from 'react';
import PropTypes from 'prop-types';
import {Image, Panel,FormGroup, FormControl} from "react-bootstrap";

export default function TextPromptViewer({state}) {
  return (
    <Panel variant="primary" style={{ height: '100%' }}>
        <Panel.Heading>
          <Panel.Title componentClass="h4">Text Prompt</Panel.Title>
        </Panel.Heading>
        <Panel.Body>
          <form>
            <FormGroup controlId="text-prompt">
              <FormControl type="text" placeholder="a floorplan of an exhibition hall" value="a floorplan of an exhibition hall"  />
            </FormGroup>
          </form>
          <Image src={state.sdImage} rounded style={{ maxWidth: '100%', maxHeight: '100%'}} />
        </Panel.Body>
    </Panel>
  )
}
