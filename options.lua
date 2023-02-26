option("core-only")
    set_default(false)
    set_showmenu(true)
    set_category("root menu")
    set_description("only include the core functionality of muda.")
option_end()

-- *******************
--
-- development options
--
-- *******************
option("dev")
    set_default(false)
    set_showmenu(true)
    set_category("root menu/dev")
    set_description("build muda example, playground and test. if you're the developer, you could enable this option.")
    add_deps("core-only")
    after_check(function(option)
        if option:dep("core-only"):enabled() then 
            option:enable(false)
        end
    end)
option_end()

function option_dev_core_only_related() 
    add_deps("core-only", "dev")
    after_check(function(option)
        if option:dep("dev"):enabled() then
            option:enable(true)
        end
        if option:dep("core-only"):enabled() then 
            option:enable(false)
        end
    end)
end

option("example")
    set_default(true)
    set_showmenu(true)
    set_category("root menu")
    set_description("build muda examples. if you want to see how to use muda, you could enable this option.")
    option_dev_core_only_related()
option_end()

option("test")
    set_default(false)
    set_showmenu(true)
    set_description("build muda test. if you're the developer, you should enable this option.")
    set_category("root menu/dev")
    option_dev_core_only_related()
option_end()

option("playground")
    set_default(false)
    set_showmenu(true)
    set_description("build muda playground. if you're the developer, you could enable this option.")
    set_category("root menu/dev")
    option_dev_core_only_related()
option_end()

option("gui-enabled")
    set_default(false)
    set_showmenu(true)
    set_category("root menu/dev")
    set_description("build for gui gallary. If you're the developer, you could enable this option for more intuitive examples.")
option_end()
-- *******************
--
-- module options
--
-- *******************
option("util")
    set_default(true)
    set_showmenu(true)
    set_description("includes <gui> <pba> modules.")
    set_category("root menu/modules")
    option_dev_core_only_related()
option_end()

option("ext")
    set_default(true)
    set_showmenu(true)
    set_description("includes <algorithm> <buffer> <blas> <composite> <thread-only> modules.")
    set_category("root menu/modules")
    option_dev_core_only_related()
option_end()


-- *******************
--
-- config macro options
--
-- *******************
option("ndebug")
    set_default(false)
    set_showmenu(true)
    set_description("shut down all muda runtime check.")
    set_category("root menu/config")
option_end()