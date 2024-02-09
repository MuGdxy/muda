import sys
import os

content = '''#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <example_common.h>
using namespace muda;

void @NAME@()
{
    example_desc("give a description to @NAME@");
}

TEST_CASE("@NAME@", "[@TAG@]")
{
    @NAME@();
}
'''
tag = "default"
name = "example_name"

if(len(sys.argv) == 1):
    print("usage: python mk.py [name:required] [tag:optional]")
    exit(0)

if(len(sys.argv) > 1):
    name = sys.argv[1]
if(len(sys.argv) > 2):
    tag = sys.argv[2]
content = content.replace("@NAME@", name)
content = content.replace("@TAG@", tag)

filename = name + ".cu"
if not os.path.exists(filename):
    f = open(filename, "w")
    f.write(content)
    f.close()
else:
    print(f"{filename} already exists!")