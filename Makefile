main: HW5

HW5: raytracer.o
	g++  -o "HW5"  ./raytracer.o
raytracer.o:
	g++ -O0 -g3 -Wall -w -c -fmessage-length=0 -MMD -MP -MF"raytracer.d" -MT"raytracer.d" -o "raytracer.o" "raytracer.cpp"
clean: 
	rm HW5
	rm raytracer.o
	rm raytracer.d
