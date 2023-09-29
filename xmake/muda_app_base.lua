-- a convenient function to create executable targets
-- kind = "gui" / "cui"
function muda_app_base(kind)
    add_deps("muda")

    if is_config("plat","linux") then
        add_cxflags("-lstdc++fs") 
        add_links("stdc++fs")
    end

    add_undefines("min","max")
    add_packages("eigen", {public = false})
    set_kind("binary")
    add_includedirs("external/", {public = false})
    add_headerfiles("src/muda/**.h","src/muda/**.inl")
    
    add_cugencodes("compute_75")
    add_links("cublas","cusparse")
end