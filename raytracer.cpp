/*
	A very basic raytracer in c++;
	Many thanks to www.scratchapixel.com for providing many examples on how to
	go about creating this

	by Tyler Decker



	Compile with the following command: c++ -o raytracer -O3 -Wall raytracer.cpp

*/

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
#include <cassert>
#include <iomanip>
#include <string>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <iterator>
#include "stdio.h"
#include "math.h"

#define EPSILON 1e-6

//vector object seems like a pretty intuitive easy soloution exampled by www.scratchapixel.com
template<typename T>
class Vec3
{
public:
	T x, y, z;
	Vec3() : x(T(0)), y(T(0)), z(T(0)) {}
	Vec3(T xx) : x(xx), y(xx), z(xx) {}
	Vec3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}
	Vec3& normalize()
	{
		T nor2 = length2();
		if (nor2 > 0) {
			T invNor = 1 / sqrt(nor2);
			x *= invNor, y *= invNor, z *= invNor;
		}
		return *this;
	}
	Vec3<T> operator * (const T &f) const { return Vec3<T>(x * f, y * f, z * f); }
	Vec3<T> operator * (const Vec3<T> &v) const { return Vec3<T>(x * v.x, y * v.y, z * v.z); }
	T dot(const Vec3<T> &v) const { return x * v.x + y * v.y + z * v.z; }
	Vec3<T> cross(const Vec3<T> &v) const {return Vec3<T>(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x); }
	Vec3<T> operator - (const Vec3<T> &v) const { return Vec3<T>(x - v.x, y - v.y, z - v.z); }
	Vec3<T> operator + (const Vec3<T> &v) const { return Vec3<T>(x + v.x, y + v.y, z + v.z); }
	Vec3<T>& operator += (const Vec3<T> &v) { x += v.x, y += v.y, z += v.z; return *this; }
	Vec3<T>& operator *= (const Vec3<T> &v) { x *= v.x, y *= v.y, z *= v.z; return *this; }
	Vec3<T> operator - () const { return Vec3<T>(-x, -y, -z); }
	bool operator == (const Vec3<T> &v) const { if(x == v.x && y == v.y && z == v.z) return true; else return false; }
	T length2() const { return x * x + y * y + z * z; }
	T length() const { return sqrt(length2()); }
	friend std::ostream & operator << (std::ostream &os, const Vec3<T> &v)
	{
		os << "[" << v.x << " " << v.y << " " << v.z << "]";
		return os;
	}
};
template<typename T>
class material
{
public:
	std::string name;
	Vec3<T> Ka;
	Vec3<T> Kd;
	Vec3<T> Ks;
	T Ns;
	T n1;
	T Tr;
	T Kr;
	T Krf;
	material(const std::string name, const Vec3<T> &ka, const Vec3<T> &kd,
			const Vec3<T> &ks, const T &ns, const T &n1, const T &tr, const T &kr, const T &krf
			) : 
			name(name), Ka(ka), Kd(kd), Ks(ks), Ns(ns), n1(n1), Tr(tr), Kr(kr), Krf(krf)
	{}
	material():
			name(), Ka(), Kd(), Ks()
	{}
};
template<typename T>
class camera
{
public:
	std::string name;
	Vec3<T> PRP;
	Vec3<T> VPN;
	Vec3<T> VUP;
	T near;
	T far;
	camera(const std::string name, const T &prp_x, const T &prp_y, const T &prp_z, const T &vpn_x, const T &vpn_y, const T &vpn_z, const T &vup_x, const T &vup_y, const T &vup_z, const T &near, const T &far
			) :
			name(name), PRP(prp_x,prp_y,prp_z), VPN(vpn_x,vpn_y,vpn_z), VUP(vup_x,vup_y,vup_z), near(near), far(far)
	{}
};
template<typename T>
class light
{
public:
	Vec3<T> center;
	T homogenius_coord;
	Vec3<T> color;
	light(const T &center_x, const T &center_y, const T &center_z, const T &center_w, const T &surface_r, const T &surface_g, const T &surface_b
			) :
			center(center_x,center_y,center_z), homogenius_coord(center_w), color(surface_r,surface_g,surface_b)
	{}
};
template<typename T>
class scene
{
public:
	std::string scene_name;
	T width;
	T height;
	T recursion;
	scene(const std::string scene_name, const T &width, const T &height, const T &recursion) :
		scene_name(scene_name), width(width), height(height), recursion(recursion)
	{}
};
template<typename T>
class Sphere
{
public:
	Vec3<T> center;                         /// position of the sphere
	T radius, radius2;                      /// sphere radius and radius^2
	material<T> type_material;					/// surface color
	Sphere(const Vec3<T> &c, const T &r, const material<T> &sc
		) :
		center(c), radius(r), radius2(r * r), type_material(sc)
	{}
	// compute a ray-sphere intersectionn
	bool intersect(const Vec3<T> &rayorig, const Vec3<T> &raydir, T *t0 = NULL, T *t1 = NULL) const
	{
		//following example Ray sphere intersection demo
		Vec3<T> c = center - rayorig; //finding the vecC value
		T v = c.dot(raydir); //already had ray dir and its normalized, this finds distance from E to V
		if (v < 0) return false;
		T d2 = c.dot(c) - v * v; //this value is (c^2 - v^2)
		if (d2 > radius2) return false; // if this value is greater than radius^2 then no intersection
		T d = sqrt(radius2 - d2);
		if (t0 != NULL && t1 != NULL) {
			*t0 = v - d;
			*t1 = v + d;
		}

		return true;
	}
};
template<typename T>
class face
{
public:
	Vec3<T> A;
	Vec3<T> B;
	Vec3<T> C;
	material<T> type_material;
	face(const Vec3<T> &a, const Vec3<T> &b, const Vec3<T> &c, const material<T> &sc
			) :
			A(a), B(b), C(c), type_material(sc)
	{}
	//compute triangle intersection
	bool intersect(const Vec3<T> &rayorig, const Vec3<T> &raydir, T *t = NULL) const
	{
		 // compute plane's normal
		Vec3<T> A_edge, B_edge, C_perp;
		A_edge = B - A;
		B_edge = C - A; // no need to normalize vector

		Vec3<T> pvec = raydir.cross(B_edge);
		T det = A_edge.dot(pvec);

		if(det > -EPSILON && det < EPSILON) return false;
		T invDet = 1 / det;

		Vec3<T> tvec = rayorig - A;
		T u = tvec.dot(pvec) * invDet;
		if(u < 0 || u > 1) return false;
		Vec3<T> qvec = tvec.cross(A_edge);
		T v = raydir.dot(qvec) * invDet;
		if(v < 0 || u + v > 1) return false;
		*t = B_edge.dot(qvec) * invDet;

		return true;
	}

	Vec3<T> normal() const {
		Vec3<T> A_edge, B_edge;
		A_edge = B - A;
		B_edge = C - A;
		return A_edge.cross(B_edge);
	}
	bool operator == (const face<T> &v) const { if(A == v.A && B == v.B && C == v.C) return true; else return false; }
};
template<typename T>
class group
{
public:
	std::string Name;
	std::vector<face<float> *> faces;
	material<T> type_material;
	group(const std::string name, const material<T> &sc) : Name(name), type_material(sc) {}
};
template<typename T>
Vec3<T> recursive_trace(const Vec3<T> &rayorig, const Vec3<T> &raydir,
	const std::vector<Sphere<T> *> &spheres, const std::vector<group<T> *> &groups,
	const std::vector<light<T> *> &lights,const T &near, const T &far, T recursion)
{
	//if (raydir.length() != 1) std::cerr << "Error " << raydir << std::endl;
	T s_tnear = INFINITY;
	T f_tnear = INFINITY;
	T tnear = INFINITY;
	const Sphere<T> *sphere = NULL;
	const face<T> *face = NULL;
	// find intersection of this ray with the sphere in the scene
	for (unsigned i = 0; i < spheres.size(); ++i) {
		T t0 = INFINITY, t1 = INFINITY;
		if (spheres[i]->intersect(rayorig, raydir, &t0, &t1)) {
			if (t0 < 0) t0 = t1;
			if (t0 < s_tnear) {
				s_tnear = t0;
				sphere = spheres[i];
			}
		}
	}
	//find intersection with faces
	for (unsigned i = 0; i < groups.size(); ++i) {
	  for (unsigned j = 0; j < groups[i]->faces.size(); ++j){
		T t = INFINITY;
		if (groups[i]->faces[j]->intersect(rayorig, raydir, &t)) {
			if (t < f_tnear) {
				f_tnear = t;
				face = groups[i]->faces[j];
			}
		}
	  }
	}

	// if there's no intersection return black or background color
	if (!sphere && !face) return Vec3<T>(0);
	if(f_tnear < s_tnear){
		tnear = f_tnear;
		sphere = NULL;
	}
	else{
		tnear = s_tnear;
		face = NULL;
	}

	// get points of intersection
	Vec3<T> intersectionPoints = rayorig + (raydir * tnear);
	Vec3<T> surfaceColor = NULL;
	if(!sphere)
	{
		surfaceColor = (face->type_material.Ka * Vec3<T>(10));
	}
	else
	{
		surfaceColor = (sphere->type_material.Ka * Vec3<T>(10));
	}

	for (unsigned i = 0; i < lights.size(); ++i){
		Vec3<T> light_direction = NULL;
		if(lights[i]->homogenius_coord == 0){
			light_direction = lights[i]->center;
		}
		else{
			light_direction = lights[i]->center - intersectionPoints;
		}
		light_direction.normalize();
		if(lighttrace(intersectionPoints,light_direction, spheres, groups, face, sphere)){
			if(!sphere){

				Vec3<T> normal = face->normal().normalize();
				if(normal.dot(light_direction) > 0) surfaceColor += face->type_material.Kd * lights[i]->color * normal.dot(light_direction);
				Vec3<T> V = (rayorig - intersectionPoints).normalize();
				Vec3<T> R = (normal * ( (light_direction.dot(normal)) * T(2.0) )) - light_direction;

				//check for specular reflection
				if(raydir.dot(R) <= 0){
					surfaceColor += face->type_material.Ks * lights[i]->color * pow(V.dot(R),face->type_material.Ns);
				}


			}
			else{

				Vec3<T> normal = (intersectionPoints - sphere->center).normalize();
				if(normal.dot(light_direction) > 0) surfaceColor +=sphere->type_material.Kd * lights[i]->color * normal.dot(light_direction);
				Vec3<T> V = (rayorig - intersectionPoints).normalize();
				Vec3<T> R = (normal * ( (light_direction.dot(normal)) * T(2.0) )) - light_direction;

				//check for specular reflection
				if(raydir.dot(R) <= 0){
					surfaceColor += sphere->type_material.Ks * lights[i]->color * pow(V.dot(R),sphere->type_material.Ns);
				}

				//recursion


				}
			}
		}

	if(!sphere)
	{
		if(recursion > 0)
		{
		Vec3<T> normal = face->normal().normalize();
		//calculate reflection
		Vec3<T> refl = raydir - (normal * (raydir.dot(normal) * T(2.0)));
						//recursive call
		refl.normalize();
		surfaceColor += recursive_trace(intersectionPoints, refl, spheres, groups, lights, near, far, recursion - 1) * face->type_material.Kr;\
		surfaceColor = (surfaceColor * face->type_material.Tr);
		}
	}
	else
	{
		if(recursion > 0)
		{
		Vec3<T> normal = (intersectionPoints - sphere->center).normalize();
		//calculate refraction
		T u = 1/sphere->type_material.n1;
		T alpha = -u;
		T beta = (u*raydir.dot(normal)) - sqrt((1 - (u*u) + (u*u) * pow(raydir.dot(normal),2)));
		Vec3<T> T_vec1 = raydir * alpha + normal * beta;
		Vec3<T> Pc = (sphere->center - intersectionPoints);
		T small_v = Pc.dot(T_vec1);
		Vec3<T> S2 = Pc + (T_vec1 * 2 *small_v);
		Vec3<T> refract_normal = (sphere->center - S2).normalize();
		u = sphere->type_material.n1/1;
		alpha = -u;
		beta = (u*T_vec1.dot(refract_normal)) - sqrt((1 - (u*u) + (u*u) * pow(T_vec1.dot(refract_normal),2)));
		Vec3<T> T_vec2 = T_vec1 * alpha + refract_normal * beta;
		Vec3<T> refracted_surfaceColor = recursive_trace(S2, T_vec2, spheres, groups, lights, near, far, recursion) * sphere->type_material.Krf;

		//calculate reflection
		Vec3<T> refl = raydir - (normal * (raydir.dot(normal) * T(2.0)));
		//recursive call
		refl.normalize();
		surfaceColor += recursive_trace(intersectionPoints, refl, spheres, groups, lights, near, far, recursion - 1) * sphere->type_material.Kr;
		surfaceColor = (surfaceColor * sphere->type_material.Tr) + (refracted_surfaceColor * (1 - sphere->type_material.Tr));

		}
	}
		return surfaceColor;

}
// This is the main trace function. It takes a ray origin and dest as arguments as well as the cameras near and far planes
template<typename T>
Vec3<T> trace(const Vec3<T> &rayorig, const Vec3<T> &raydir,
	const std::vector<Sphere<T> *> &spheres, const std::vector<group<T> *> &groups,
	const std::vector<light<T> *> &lights, const T &near, const T &far, const T &recursion)
{
	//if (raydir.length() != 1) std::cerr << "Error " << raydir << std::endl;
	T s_tnear = INFINITY;
	T f_tnear = INFINITY;
	T tnear = INFINITY;
	const Sphere<T> *sphere = NULL;
	const face<T> *face = NULL;
	// find intersection of this ray with the sphere in the scene
	for (unsigned i = 0; i < spheres.size(); ++i) {
		T t0 = INFINITY, t1 = INFINITY;
		if (spheres[i]->intersect(rayorig, raydir, &t0, &t1)) {
			if (t0 < 0) t0 = t1;
			if (t0 < s_tnear) {
				s_tnear = t0;
				sphere = spheres[i];
			}
		}
	}
	//find intersection with faces
	for (unsigned i = 0; i < groups.size(); ++i) {
	  for (unsigned j = 0; j < groups[i]->faces.size(); ++j){
		T t = INFINITY;
		if (groups[i]->faces[j]->intersect(rayorig, raydir, &t)) {
			if (t < f_tnear) {
				f_tnear = t;
				face = groups[i]->faces[j];
			}
		}
	  }
	}

	// if there's no intersection return black or background color
	if (!sphere && !face) return Vec3<T>(0);
	if(f_tnear < s_tnear){
		tnear = f_tnear;
		sphere = NULL;
	}
	else{
		tnear = s_tnear;
		face = NULL;
	}
	if(tnear > far) return Vec3<T>(0);
	if(tnear < near) return Vec3<T>(0);

	// get points of intersection
	Vec3<T> intersectionPoints = rayorig + (raydir * tnear);
	Vec3<T> surfaceColor = NULL;
	if(!sphere)
	{
		surfaceColor = (face->type_material.Ka * Vec3<T>(10));
	}
	else
	{
		surfaceColor = (sphere->type_material.Ka * Vec3<T>(10));
	}
	for (unsigned i = 0; i < lights.size(); ++i){
		Vec3<T> light_direction = NULL;
		if(lights[i]->homogenius_coord == 0){
			light_direction = lights[i]->center;
		}
		else{
			light_direction = lights[i]->center - intersectionPoints;
		}
		light_direction.normalize();
		if(lighttrace(intersectionPoints,light_direction, spheres, groups, face, sphere)){
			if(!sphere){

				Vec3<T> normal = face->normal().normalize();
				if(normal.dot(light_direction) > 0) surfaceColor += face->type_material.Kd * lights[i]->color * normal.dot(light_direction);
				Vec3<T> V = (rayorig - intersectionPoints).normalize();
				Vec3<T> R = (normal * ( (light_direction.dot(normal)) * T(2.0) )) - light_direction;

				//check for specular reflection
				if(raydir.dot(R) <= 0){
					surfaceColor += face->type_material.Ks * lights[i]->color * pow(V.dot(R),face->type_material.Ns);
				}

			}
			else{

				Vec3<T> normal = (intersectionPoints - sphere->center).normalize();
				if(normal.dot(light_direction) > 0) surfaceColor +=  sphere->type_material.Kd * lights[i]->color * normal.dot(light_direction);
				Vec3<T> V = (rayorig - intersectionPoints).normalize();
				Vec3<T> R = (normal * ( (light_direction.dot(normal)) * T(2.0) )) - light_direction;

				//check for specular reflection
				if(raydir.dot(R) <= 0){
					surfaceColor += sphere->type_material.Ks * lights[i]->color * pow(V.dot(R),sphere->type_material.Ns);
				}

			}
		}

	}
	if(!sphere)
	{
		Vec3<T> normal = face->normal().normalize();
		//calculate reflection
		Vec3<T> refl = raydir - (normal * (raydir.dot(normal) * T(2.0)));
						//recursive call
		refl.normalize();
		surfaceColor += recursive_trace(intersectionPoints, refl, spheres, groups, lights, near, far, recursion - 1) * face->type_material.Kr;
		surfaceColor = (surfaceColor * face->type_material.Tr);
	}
	else
	{
		Vec3<T> normal = (intersectionPoints - sphere->center).normalize();
		//calculate refraction
		T u = 1/sphere->type_material.n1;
		T alpha = -u;
		T beta = (u*raydir.dot(normal)) - sqrt((1 - (u*u) + (u*u) * pow(raydir.dot(normal),2)));
		Vec3<T> T_vec1 = raydir * alpha + normal * beta;
		Vec3<T> Pc = (sphere->center - intersectionPoints);
		T small_v = Pc.dot(T_vec1);
		Vec3<T> S2 = Pc + (T_vec1 * 2 * small_v);
		Vec3<T> refract_normal = (sphere->center - S2).normalize();
		u = sphere->type_material.n1/1;
		alpha = -u;
		beta = (u*T_vec1.dot(refract_normal)) - sqrt((1 - (u*u) + (u*u) * pow(T_vec1.dot(refract_normal),2)));
		Vec3<T> T_vec2 = T_vec1 * alpha + refract_normal * beta;
		Vec3<T> refracted_surfaceColor = recursive_trace(S2, T_vec2, spheres, groups, lights, near, far, recursion) * sphere->type_material.Krf;

		//calculate reflection
		Vec3<T> refl = raydir - (normal * (raydir.dot(normal) * T(2.0)));
		//recursive call
		refl.normalize();
		surfaceColor += recursive_trace(intersectionPoints, refl, spheres, groups, lights, near, far, recursion - 1) * sphere->type_material.Kr;
		surfaceColor = (surfaceColor * sphere->type_material.Tr) + (refracted_surfaceColor * (1 - sphere->type_material.Tr));
	}

	return surfaceColor;
}
template<typename T>
bool lighttrace(const Vec3<T> &rayorig, const Vec3<T> &raydir,
	const std::vector<Sphere<T> *> &spheres, const std::vector<group<T> *> &groups, const face<T> *the_face, const Sphere<T> *the_sphere)
{
	//if (raydir.length() != 1) std::cerr << "Error " << raydir << std::endl;
	const face<T> *otherface;
	T tnear = INFINITY;
	const Sphere<T> *sphere = NULL;
	const face<T> *face = NULL;



	// find intersection of this ray with the sphere in the scene
	for (unsigned i = 0; i < spheres.size(); ++i) {
		T t0 = INFINITY, t1 = INFINITY;
		if (spheres[i]->intersect(rayorig, raydir, &t0, &t1)) {
			if (t0 < 0) t0 = t1;
			if (t0 > 0 ) {
				tnear = t0;
				if(tnear > 0) sphere = spheres[i];
			}
		}
	}
	if(!the_face){
		 for (unsigned i = 0; i < groups.size(); ++i) {
		  for (unsigned j = 0; j < groups[i]->faces.size(); ++j){
			T t = INFINITY;
			if (groups[i]->faces[j]->intersect(rayorig, raydir, &t)) {
				if (t > 0) {
					tnear = t;
					face = groups[i]->faces[j];
				}
			}
		  }
		 }
		}
	else{
		for (unsigned i = 0; i < groups.size(); ++i) {
		 for (unsigned j = 0; j < groups[i]->faces.size(); ++j){
			T t = INFINITY;
			otherface = groups[i]->faces[j];
			if(*the_face == *otherface){
				otherface = NULL;
			}
			else if (groups[i]->faces[j]->intersect(rayorig, raydir, &t)) {
				if(t > 0){
					tnear = t;
					face = groups[i]->faces[j];
				}
			}
		  }
		 }
		}

	// if there's no intersection return true
	if (!sphere && !face) return true;
	else return false;

}

// Main rendering function. compute a camera ray for each pixel of the image
// trace it and return a color. If the ray hits a sphere, we return the color of the
// sphere at the intersection point, else we return the background color.
template<typename T>
void render(const std::vector<Sphere<T> *> &spheres, const std::vector<light<T> *> &lights, const std::vector<camera<T> *> &cameras,
		const scene<T> &the_scene, const std::vector<group<T> *> &groups, const T &recursion)
{


		// find intersection of this ray with the sphere in the scene
	for (unsigned i = 0; i < cameras.size(); ++i) {
		Vec3<T>  n_vector = cameras[i]->VPN.normalize();
		Vec3<T>  u_vector = cameras[i]->VUP.cross(n_vector).normalize();
		Vec3<T>  v_vector = n_vector.cross(u_vector);

		unsigned width = (unsigned)the_scene.width;
		unsigned height = (unsigned)the_scene.height;
		Vec3<T> *c_image = new Vec3<T>[width * height], *c_pixel = c_image;

		T invWidth = 1 / T(width), invHeight = 1 / T(height);


		T near = cameras[i]->near;
		T far = cameras[i]->far;

	// Trace rays
		for (unsigned y = 0; y < height; ++y) {
			for (unsigned x = 0; x < width; ++x, ++c_pixel) {
				T xx = (2 * ((x + 0.5) * invWidth) - 1);
				T yy = (1 - 2 * ((y + 0.5) * invHeight));
				Vec3<T> raydir = (-(cameras[i]->VPN * near) +   u_vector * xx +  v_vector * yy );
				Vec3<T> pixel_point;
				raydir.normalize();
				*c_pixel = trace(cameras[i]->PRP, raydir, spheres, groups, lights, near, far, recursion);
			}
		}
		std::ostringstream oss;
		oss << the_scene.scene_name <<"_" << cameras[i]->name <<"_color.ppm";
		char * file =  new char[oss.str().length() + 1];
		std::strcpy(file,oss.str().c_str());
	// Save result to a PPM image
		std::ofstream ofs(file, std::ios::out | std::ios::binary);
		ofs << "P6\n" << width << " " << height << "\n255\n";
		for (unsigned j = 0; j < width * height; ++j) {
			ofs << (unsigned char)(std::min(T(255), c_image[j].x)) <<
					(unsigned char)(std::min(T(255), c_image[j].y)) <<
					(unsigned char)(std::min(T(255), c_image[j].z));
		}
		ofs.close();
		oss.str("");
		oss.clear();

		delete [] c_image;

	}
}
char * strCut( char *s, const char *pattern )
{
   if ( char *p = std::strstr( s, pattern ) )
   {
      char *q = p + std::strlen( pattern );
      while ( *p++ = *q++ );
   }

   return s;
}
using namespace std;
int main(int argc, char **argv)
{
	srand48(13);
	std::vector<Sphere<float> *> spheres;
	std::vector<camera<float> *> cameras;
	std::vector<scene<float> *> scenes;
	std::vector<material<float> *> materials;
	std::vector<light<float> *> lights;
	std::vector<Vec3<float> *> vertices;
	std::vector<group<float> *> groups;

	//Test
/*
 * test
	//scene
	scenes.push_back(new scene<float>("r", "scene_01", 512, 512, 1));
	//PRP VPN near and far
	cameras.push_back(new camera<float>("cam00",0,0,0,0,0,1,4,13.5));
	cameras.push_back(new camera<float>("cam01",0,0,0,0,0,1,8,15));
	// position, radius, surface color
	spheres.push_back(new Sphere<float>(Vec3<float>(-1,  1, -8), 1, Vec3<float>(255, 0, 0)));
	spheres.push_back(new Sphere<float>(Vec3<float>(1,  1, -10), 1, Vec3<float>(0, 255, 0)));
	spheres.push_back(new Sphere<float>(Vec3<float>(1, -1, -12), 1, Vec3<float>(0, 0, 255)));
	spheres.push_back(new Sphere<float>(Vec3<float>(-1, -1, -14), 1, Vec3<float>(0, 255, 255)));
*/

	string STRING1;
	ifstream infile1;
	infile1.open (argv[2]);
	        while(getline(infile1,STRING1)) // To get you all the lines.
	        {

	        	vector<string> tokens;
	        	istringstream iss(STRING1);
	        	copy(istream_iterator<string>(iss),
	        	         istream_iterator<string>(),
	        	         back_inserter(tokens));
	        	if(tokens.empty()) continue;
	        	if(tokens[0].compare("c") == 0){
	        		cameras.push_back(new camera<float>(tokens[1],strtof(tokens[2].c_str(),NULL),strtof(tokens[3].c_str(),NULL),
	        				strtof(tokens[4].c_str(),NULL),strtof(tokens[5].c_str(),NULL),strtof(tokens[6].c_str(),NULL),
	        				strtof(tokens[7].c_str(),NULL),strtof(tokens[8].c_str(),NULL),strtof(tokens[9].c_str(),NULL),
	        				strtof(tokens[10].c_str(),NULL),strtof(tokens[11].c_str(),NULL),strtof(tokens[12].c_str(),NULL)));
	        	}
	        	else if(tokens[0].compare("r") == 0){
	        		scenes.push_back(new scene<float>(tokens[1], strtof(tokens[2].c_str(),NULL), strtof(tokens[3].c_str(),NULL), strtof(tokens[4].c_str(),NULL)));
	        	}
	        	else if(tokens[0].compare("l") == 0){
	        		lights.push_back(new light<float>(strtof(tokens[1].c_str(),NULL), strtof(tokens[2].c_str(),NULL), strtof(tokens[3].c_str(),NULL), strtof(tokens[4].c_str(),NULL),
	        						strtof(tokens[5].c_str(),NULL), strtof(tokens[6].c_str(),NULL), strtof(tokens[7].c_str(),NULL)));
	        	}
	        }
	infile1.close();
	string ffile = argv[1];
	char * path = strCut(argv[1], "model.obj");
	string STRING2;
	ifstream infile2;
	infile2.open (ffile.c_str());
				group<float> *latestgroup;
				string name_material;
		        while(getline(infile2,STRING2)) // To get you all the lines.
		        {

		        	vector<string> tokens;
		        	istringstream iss(STRING2);
		        	copy(istream_iterator<string>(iss),
		        	         istream_iterator<string>(),
		        	         back_inserter(tokens));
		        	if(tokens.empty()) continue;
		        	if(tokens[0].compare("usemtl") == 0){
		        		name_material = tokens[1];
		        	}
		        	else if(tokens[0].compare("v") == 0){
		        		   vertices.push_back(new Vec3<float>(strtof(tokens[1].c_str(),NULL), strtof(tokens[2].c_str(),NULL), strtof(tokens[3].c_str(),NULL)));
		        	}
		        	else if(tokens[0].compare("s") == 0){
		        		material<float> the_material;
		        		for (unsigned i = 0; i < materials.size(); ++i){
		        			if(materials[i]->name.compare(name_material) == 0){
		        				the_material = *materials[i];
		        			}
		        		}
		        		spheres.push_back(new Sphere<float>(Vec3<float>(strtof(tokens[2].c_str(),NULL),  strtof(tokens[3].c_str(),NULL), strtof(tokens[4].c_str(),NULL)),
		        				strtof(tokens[5].c_str(),NULL), the_material
		        				));

		        	}
		        	else if (tokens[0].compare("g") == 0){
		        		material<float> the_material;
		        				        		for (unsigned i = 0; i < materials.size(); ++i){
		        				        			if(materials[i]->name.compare(name_material) == 0){
		        				        				the_material = *materials[i];
		        				        			}
		        				        		}
		        		latestgroup = new group<float>(tokens[1],the_material);
		        		groups.push_back(latestgroup);

		        	}
		        	else if (tokens[0].compare("f") == 0){
		        		material<float> the_material;
		        				        		for (unsigned i = 0; i < materials.size(); ++i){
		        				        			if(materials[i]->name.compare(name_material) == 0){
		        				        				the_material = *materials[i];
		        				        			}
		        				        		}
		        		if(tokens.size() > 4){
		        			for ( int i = 1; (i + 2) < tokens.size(); i++){
		        				latestgroup->faces.push_back(new face<float>(*vertices[atoi(tokens[1].c_str()) - 1], *vertices[atoi(tokens[1 + i].c_str()) - 1], *vertices[atoi(tokens[1 + (i + 1)].c_str()) - 1], the_material ));
		        			}
		        		}
		        		else{
		        			latestgroup->faces.push_back(new face<float>(*vertices[atoi(tokens[1].c_str()) - 1], *vertices[atoi(tokens[2].c_str()) - 1], *vertices[atoi(tokens[3].c_str()) - 1], the_material  ));
		        		}

		        	}
		        	else if(tokens[0].compare("mtllib") == 0){
		        		string STRING3;
		        		ifstream infile3;
		        		string path_a = path;
		        		string the_string = path_a +  tokens[1].c_str();
		        		infile3.open (the_string.c_str());

						string name;
						Vec3<float> Ka;
						Vec3<float> Kd;
						Vec3<float> Ks;
						float Ns;
						float n1;
						float Tr;
						float Kr;
						float Krf;
						int filled = 0;

		        		while(getline(infile3,STRING3)) // To get you all the lines.
		        				        {

		        				        	vector<string> tokens_in;
		        				        	istringstream iss_in(STRING3);
		        				        	copy(istream_iterator<string>(iss_in),
		        				        	         istream_iterator<string>(),
		        				        	         back_inserter(tokens_in));
		        				        	if(tokens_in.empty()) continue;
		        				        	if(tokens_in[0].compare("newmtl") == 0){
		        				        		name = tokens_in[1];
		        				        		filled ++;

		        				        	}
		        				        	else if(tokens_in[0].compare("Ka") == 0){
		        				        		Ka = Vec3<float>(strtof(tokens_in[1].c_str(),NULL),  strtof(tokens_in[2].c_str(),NULL), strtof(tokens_in[3].c_str(),NULL));
		        				        		filled ++;
		        				        	}
		        				        	else if(tokens_in[0].compare("Kd") == 0){
		        				        		Kd = Vec3<float>(strtof(tokens_in[1].c_str(),NULL),  strtof(tokens_in[2].c_str(),NULL), strtof(tokens_in[3].c_str(),NULL));
		        				        		filled ++;
		        				        	}
		        				        	else if(tokens_in[0].compare("Ks") == 0){
		        				        		Ks = Vec3<float>(strtof(tokens_in[1].c_str(),NULL),  strtof(tokens_in[2].c_str(),NULL), strtof(tokens_in[3].c_str(),NULL));
		        				        		filled ++;
		        				        	}
		        				        	else if(tokens_in[0].compare("Ns") == 0){
		        				        		Ns = strtof(tokens_in[1].c_str(),NULL);
		        				        		filled ++;
		        				        	}
		        				        	else if(tokens_in[0].compare("n1") == 0){
		        				        		n1 = strtof(tokens_in[1].c_str(),NULL);
		        				        		filled ++;
		        				        	}
		        				        	else if(tokens_in[0].compare("Tr") == 0){
		        				        		Tr = strtof(tokens_in[1].c_str(),NULL);
		        				        		filled ++;
		        				        	}
		        				        	else if(tokens_in[0].compare("Kr") == 0){
		        				        		Kr = strtof(tokens_in[1].c_str(),NULL);
		        				        		filled ++;
		        				        	}
		        				        	else if(tokens_in[0].compare("Krf") == 0){
		        				        		Krf = strtof(tokens_in[1].c_str(),NULL);
		        				        		filled ++;
		        				        	}
		        				        	if(filled == 9){
		        				        		materials.push_back(new material<float>(name,Ka,Kd,Ks,Ns,n1,Tr,Kr,Krf));
		        				        		name = "";
		        				        		Ka = NULL;
		        				        		Kd = NULL;
		        				        		Ks = NULL;
		        				        		Ns = NULL;
		        				        		n1 = NULL;
		        				        		Tr = NULL;
		        				        		Kr = NULL;
		        				        		Krf = NULL;
		        				        		filled = 0;
		        				        	}


		        				        }
		        		infile3.close();
		        	}

		        }
	infile2.close();


	for (unsigned i = 0; i < scenes.size(); ++i) {
		scene<float> the_scene = *scenes[i];
		render<float>(spheres, lights, cameras, the_scene, groups, the_scene.recursion);
	}

	while (!spheres.empty()) {
		Sphere<float> *sph = spheres.back();
		spheres.pop_back();
		delete sph;
	}
	while (!lights.empty()) {
		light<float> *sph = lights.back();
		lights.pop_back();
		delete sph;
	}
	while (!groups.empty()) {
		group<float> *sph = groups.back();
		groups.pop_back();
		delete sph;
	}
	while (!vertices.empty()) {
		Vec3<float> *sph = vertices.back();
		vertices.pop_back();
		delete sph;
	}
	while (!cameras.empty()) {
		camera<float> *sph = cameras.back();
		cameras.pop_back();
		delete sph;
	}
	while (!scenes.empty()) {
		scene<float> *sph = scenes.back();
		scenes.pop_back();
		delete sph;
	}

	return 0;
}
