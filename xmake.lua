set_project("muda")

add_rules("mode.debug", "mode.release")
set_languages("cxx17")

add_requires("cuda", {optional = false})

option("eigen_dir")
    set_default("default")
    set_showmenu(true)
    set_description("user defined eigen directory. if you want to use your own eigen, you should set this option to your eigen directory")
option_end()

target("muda")
    add_undefines("min","max",{public = true})
    set_kind("headeronly")
    add_headerfiles("src/muda/**.h","src/muda/**.inl")
    add_includedirs("src/", {public = true})

    if(is_config("eigen_dir", "default")) then
        add_headerfiles("external/default/**")
        add_includedirs("external/default/", {public = true})
    else
        add_includedirs(get_config("eigen_dir"), {public = true})
    end

    add_cuflags("--extended-lambda", {public = true}) -- must be set for muda paradigm
    add_cuflags("--expt-relaxed-constexpr", {public = true})

option("test")
    set_showmenu(true)
    set_description("build muda test. if you're the developer, you should enable this option")
option_end()

if has_config("test") then
    target("muda_test")
        add_undefines("min","max",{public = true})
        set_kind("binary")
        add_includedirs("test/", {public = false})
        add_headerfiles("src/muda/**.h","src/muda/**.inl")
        add_files("test/muda_test/**.cu","test/muda_test/**.cpp")
        add_cugencodes("native")
        add_cugencodes("compute_75")
        add_links("cublas","cusparse")
        add_deps("muda")
end


-- examples
option("example")
    set_default(false)
    set_showmenu(true)
    set_description("build muda examples. if you want to see how to use muda, you could enable this option")
option_end()

if has_config("example") then
    target("muda_example")
        add_undefines("min","max",{public = true})
        set_kind("binary")
        add_includedirs("test/", {public = false})
        add_headerfiles("src/muda/**.h","src/muda/**.inl")
        add_files("example/**.cu","example/**.cpp")
        add_cugencodes("native")
        add_cugencodes("compute_75")
        add_deps("muda")
end
