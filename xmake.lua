set_project("muda")
-- to keep this "xmake.lua" as concise as possible
-- we put all the options to "options.lua"
-- and include it here.
includes("options.lua")

-- **********************************
-- 
-- packages requirement
-- 
-- **********************************


add_requires("cuda", {optional = false})
add_requires("eigen", {optional = false})

if (has_config("gui-enabled")) then 
    add_requires("glfw", {optional = true})
end

-- **********************************
-- 
-- targets
-- 
-- **********************************
set_languages("cxx17")
add_rules("mode.debug", "mode.release")


target("muda")
    add_undefines("min","max")
    set_kind("headeronly")
    add_headerfiles("src/core/(muda/**.h)","src/core/(muda/**.inl)",{public = true})
    add_includedirs("src/core/", {public = true})
    if(has_config("ndebug")) then
        add_defines("MUDA_NDEBUG=1", {public = true})
    end
    add_packages("cuda", "eigen", {public = true})
    add_cuflags("--extended-lambda", {public = true}) -- must be set for muda
    add_cuflags("--expt-relaxed-constexpr", {public = true}) -- must be set for muda
target_end()


if(has_config("ext")) then
    target("muda-ext")
        add_deps("muda")
        set_kind("headeronly")
        add_includedirs("src/ext/",{public=true})
        add_headerfiles("src/ext/(muda/**.h)","src/ext/(muda/**.inl)")
        
        -- definitions for EASTL
        add_defines("_CHAR16T")
        add_defines("_CRT_SECURE_NO_WARNINGS")
        add_defines("_SCL_SECURE_NO_WARNINGS")
        add_defines("EASTL_OPENSOURCE=1")
        add_includedirs("src/ext/muda/thread_only", {public = true}) -- EASTL requirement
        add_includedirs("src/ext/muda/thread_only/EABase/include/Common", {public = true}) -- EASTL requirement
    target_end()
end 

if(has_config("util")) then
    -- this target includes Physically-Based-Animation algorithms
    target("muda-pba")
        add_deps("muda-ext")
        set_kind("headeronly")
        add_includedirs("src/util/", {public=true})
        add_headerfiles("src/util/(muda/pba/**.h)","src/util/(muda/pba/**.inl)")
    target_end()  

    -- TODO: need linux gui fix 
    if (has_config("gui-enabled")) then 
        -- this target includes GUI
        target("muda-gui")
            add_deps("muda-ext")
            add_packages("glfw", {public = true})
            set_kind("static")
            -- add imgui src
            add_headerfiles("external/(imgui/**.h)")
            add_files("external/imgui/**.cpp")
            add_includedirs("external/", {public=true})
            -- add muda gui src
            add_files(
                "src/util/muda/gui/**.cpp",
                "src/util/muda/gui/**.cu")
            add_includedirs("src/util/", {public=true})
            add_headerfiles("src/util/(muda/gui/**.h)")
        target_end()
    end
end

-- TODO: need linux gui fix 
if(has_config("util") 
    and has_config("ext")) then
    -- this is a phony target to collect all muda functionalities which is convenient for quick-starts, examples and tests.
    target("muda-full")
        add_deps(
            "muda-ext",
            "muda-pba"
        )
        if (has_config("gui-enabled")) then 
            add_deps("muda-gui")
        end 

        set_kind("phony")
    target_end()
end


-- a convenient function to create executable targets
-- kind = "gui" / "cui"
function muda_app_base(kind)
    if(kind == "gui") then 
        add_deps("muda-full")
    elseif (kind == "cui") then
        add_deps("muda-ext","muda-pba")
    end
    
    if is_config("plat","linux") then
        add_cxflags("-lstdc++fs") 
        add_links("stdc++fs")
    end

    add_undefines("min","max")
    
    set_kind("binary")
    add_includedirs("test/", {public = false})
    add_headerfiles("src/core/muda/**.h","src/core/muda/**.inl")
    add_headerfiles("src/ext/muda/**.h","src/ext/muda/**.inl")
    add_headerfiles("src/util/muda/**.h","src/util/muda/**.inl")
    
    add_cugencodes("native")
    add_cugencodes("compute_75")
    add_links("cublas","cusparse")
end

if has_config("test") then
    target("muda_test")
        muda_app_base("cui")
        test_data_dir = path.absolute("test/data")
        add_defines("MUDA_TEST_DATA_DIR=R\"(".. test_data_dir..")\"")
        add_files("test/muda_test/**.cu","test/muda_test/**.cpp")
    target_end()
end

if has_config("example") then
    target("muda_example")
        muda_app_base("cui")
        add_files("example/**.cu","example/**.cpp")
end

if has_config("playground") then
    target("muda_pg")
        if (has_config("gui-enabled")) then 
            muda_app_base("gui")
            add_files("test/playground/**.cu","test/playground/**.cpp")
        end
    target_end()
end

