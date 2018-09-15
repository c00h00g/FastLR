CXX=g++
CXXFLAGS=-std=c++11 -g -march=native
RM=rm -f

fastlr: model.o utils.o fastlr.o
	$(CXX) $(CXXFLAGS) -o $@ $^

fastlr.o: fastlr.cc
	$(CXX) $(CXXFLAGS) -c $<

model.o: model.cc model.h
	$(CXX) $(CXXFLAGS) -c $<

utils.o: utils.cc utils.h
	$(CXX) $(CXXFLAGS) -c $<

clean:
	$(RM) *.o fastlr
