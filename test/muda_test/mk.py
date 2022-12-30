import sys

content = '''
#include <catch2/catch.hpp>
#include <muda/muda.h>

using namespace muda;

void @TEST_NAME@()
{

}

TEST_CASE("@TEST_NAME@", "[@TAG@]")
{
    @TEST_NAME@();
}
'''
tag = "default"
test_name = "test"
if(len(sys.argv) > 1):
    test_name = sys.argv[1]
if(len(sys.argv) > 2):
    tag = sys.argv[2]
content = content.replace("@TEST_NAME@", test_name)
content = content.replace("@TAG@", tag)

f = open(test_name + ".cu", "w")
f.write(content)
f.close()