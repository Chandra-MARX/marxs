

THREE.MARXSLoader = function( manager ) {

    this.manager = ( manager !== undefined ) ? manager : THREE.DefaultLoadingManager;
    
    this.withCredentials = false;

    this.load = function( url, onLoad, onProgress, onError ) {

		var scope = this;

		var loader = new THREE.XHRLoader( this.manager );
		loader.setWithCredentials( this.withCredentials );
		loader.load( url, function ( text ) {

		    var json = JSON.parse( text );
	            var objectlist = scope.parse( json );
		    onLoad(objectlist);
		}, onProgress, onError );
    };

    this.parse =  function ( json) {
	objects = []

	for (elem in json.elements)
	    if (elem.geometry == "BufferGeometry"){
		objects.push(parseBuffer(elem));
	    } else {
		objects.push(parsegeneral(elem));
	    }
	return objects;
    }
    

    function parsegeneral(element){
	var meshes = []
	for ( i = 1; i<element.n; i++){
	    var geometry = new THREE[element.geometry]( element.geometrypars );
	    var material = new THREE[element.material]( element.materialproperties );
	    var mesh = new THREE.Mesh( geometry, material );
	    mesh.matrixAutoUpdate = false;
	    mesh.matrix.set(element.pos4d[i]);
	    mesh.name = element.name;
	    meshes[i] = mesh;
	}
	return meshes;
    }

	
    function parseBuffer(element) {
	meshes = []
	for ( i = 0; i < element.n; i ++ ){
	    
	    var geometry = new THREE.BufferGeometry();
	    var material = new THREE[element.material]( element.materialproperties);
	    var positions = new Float32Array(element.pos[i]);
	    var colors = new Float32Array(element.color[i]);
	    geometry.addAttribute( 'position', new THREE.BufferAttribute( positions, 3 ) );
	    geometry.addAttribute( 'color', new THREE.BufferAttribute( colors, 3 ) );
	    geometry.computeBoundingSphere();
	    obj = new THREE.Line( geometry, material );
	    obj.name = element.name
	    meshes[i] = obj
	};
	return meshes;
    }
    
};

