import React from 'react';
import PropTypes from 'prop-types';
import {FaFolderOpen as IconLoad} from 'react-icons/fa';
import ToolbarButton from './toolbar-button';
import {browserImageUpload,sendPostRequest}  from '../../utils/browser';

export default function ToolbarNewProjectButton({state}, {translator, projectActions}) {
  return (
    <ToolbarButton active={false} tooltip={translator.t('New project')} onClick={projectActions.newProject()}>
      <IconLoad />
    </ToolbarButton>
  );
}
