CXX=g++
CXXFLAGS=-std=c++11 -g
RM=rm -f

model.o: model.cc model.h
	$(CXX) $(CXXFLAGS) -c $<

utils.o: utils.cc utils.h
	$(CXX) $(CXXFLAGS) -c $<

clean:
	$(RM) *.o
