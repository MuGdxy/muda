//#pragma once
//#include <muda/viewer/viewer_base_accessor.h>
//#include <muda/ext/field/field_entry_viewer.h>
//#include <muda/viewer/dense.h>
//namespace muda
//{
//class FieldViewer : public ViewerBase
//{
//    MUDA_VIEWER_COMMON_NAME(FieldViewer);
//
//  private:
//    friend class Field;
//    Dense1D<FieldEntryViewerBase> m_entries;
//    MUDA_GENERIC FieldViewer(const Dense1D<FieldEntryViewerBase>& m)
//        : m_entries(m)
//    {
//    }
//
//  public:
//    template <typename T, FieldEntryLayout Layout, int M, int N>
//    MUDA_DEVICE FieldEntryViewer<T, Layout, M, N> entry(const char* name)
//    {
//        using Viewer = FieldEntryViewer<T, Layout, M, N>;
//
//        auto strcmp = [] MUDA_DEVICE(const char* a, const char* b) -> bool
//        {
//            while(*a && *b && *a == *b)
//            {
//                a++;
//                b++;
//            }
//            return *a == *b;
//        };
//        for(int i = 0; i < m_entries.total_size(); i++)
//        {
//            auto& e = m_entries(i);
//            if(strcmp(e.name(), name))
//            {
//                MUDA_KERNEL_ASSERT(e.elem_byte_size() == sizeof(T),
//                                   "FieldViewer[%s:%s]: FieldEntry[%s] Type size mismatching, entry type size=%d, your size=%d",
//                                   kernel_name(),
//                                   this->name(),
//                                   name,
//                                   e.elem_byte_size(),
//                                   sizeof(T));
//#if MUDA_CHECK_ON
//                using Acc           = details::ViewerBaseAccessor;
//                Acc::kernel_name(e) = Acc::kernel_name(*this);
//#endif
//                return (Viewer&)e;
//            }
//        }
//        MUDA_KERNEL_ERROR_WITH_LOCATION("FieldViewer[%s:%s] FieldEntry[%s] not found",
//                                        kernel_name(),
//                                        this->name(),
//                                        name);
//        return Viewer{};
//    }
//};
//}  // namespace muda