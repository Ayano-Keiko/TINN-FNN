source := *.cpp *.h

main : $(source)
	g++ $(source) -o main

.PHONY: clean
clean :
	rm ./main
