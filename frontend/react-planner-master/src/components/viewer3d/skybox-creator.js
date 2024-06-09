import * as THREE from 'three';

export default function createSkyBox(scene) {
  const loader = new THREE.CubeTextureLoader();
  const texture = loader.load([
            require("../../../demo/dist/catalog/skybox/top/right.jpg"),
            require("../../../demo/dist/catalog/skybox/top/left.jpg"),
            require("../../../demo/dist/catalog/skybox/top/top.jpg"),
            require("../../../demo/dist/catalog/skybox/top/bottom.jpg"),
            require("../../../demo/dist/catalog/skybox/top/front.jpg"),
            require("../../../demo/dist/catalog/skybox/top/back.jpg")
        ]);
  return texture;
}
