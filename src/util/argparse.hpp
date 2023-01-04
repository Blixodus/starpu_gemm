#include <string_view>
#include <algorithm>


char* getArg(int argc, char** argv, std::string_view arg) {
	char** begin = argv;
	char** end = argv + argc;
	char** itr = std::find(begin, end, arg);

	if ((itr != end) && (itr++ != end)) {
		return *itr;
	}

	return 0;
}

bool hasArg(int argc, char** argv, std::string_view arg) {
	char** begin = argv;
	char** end = argv + argc;
	return (std::find(begin, end, arg) != end);
}
