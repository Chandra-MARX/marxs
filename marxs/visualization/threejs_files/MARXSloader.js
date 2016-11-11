
/* wrapper factory

The current version of javascript for not have an equivalent of python unpacking
where an array of arguments can be used to pass independent arguments to a function.
In normal functions "apply" can do that, but not in constructors (with "new").
This is a work-around that I found on stackoverflow.
*/
function construct(constructor, args) {
    function F() {
        return constructor.apply(this, args);
    }
    F.prototype = constructor.prototype;
    return new F();
}

/*
author: Moritz Guenther (github.com/hamogu)

Loader for THREE.js that reads MARXS json output
Ray-tracing simulations with MARXS can have hundreds of optical elements such as
gratings, CCD detectors etc. 
MARXS can write the specifications to plot this up in three.js to json file.
This loader parses the json file and constructs the appropriate three.js objects.
*/
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

	for (e in json.elements){
	    elem = json.elements[e]
	    if (elem.geometry == "BufferGeometry"){
		objects.push(parseBuffer(elem));
	    } else {
		objects.push(parsegeneral(elem));
	    }
        }
	return {'objects': objects, 'json': json};
    }
    

    function parsegeneral(element){
	var meshes = []
	for ( i = 0; i<element.n; i++){
	    var geometry = construct(THREE[element.geometry], element.geometrypars );
	    var material = new THREE[element.material]( element.materialproperties );
	    var mesh = new THREE.Mesh( geometry, material );
	    mesh.matrixAutoUpdate = false;
	    var typedpos4d = new Float32Array(element.pos4d[i]);
	    mesh.matrix.elements = typedpos4d;
	    mesh.updateMatrix();
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
	    geometry.addAttribute( 'position', new THREE.BufferAttribute( positions, 3 ) );
	    if (element.color != undefined){
		var colors = new Float32Array(element.color[i]);
		geometry.addAttribute( 'color', new THREE.BufferAttribute( colors, 3 ) );
	    }
	    if (element.faces != undefined){
		var faces = new Uint16Array(element.faces[i]);
		// itemSize = 3 because there are 3 values (components) per triangle
		geometry.setIndex(new THREE.BufferAttribute( faces, 1 ) );
	    }
	    
	    geometry.computeBoundingSphere();
	    obj = new THREE[element.geometrytype]( geometry, material );
	    obj.name = element.name
	    meshes[i] = obj
	};
	return meshes;
    }
    
};

