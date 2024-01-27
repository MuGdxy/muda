//#pragma once
//#include <muda/ext/field/field_entry_viewer.h>
//#include <muda/type_traits/type_modifier.h>
//namespace muda::eigen
//{
//template <typename T, FieldEntryLayout Layout, int M, int N>
//class CMatrixEntryViewer : public CFieldEntryViewer<T, Layout, M, N>
//{
//  public:
//    using Base = muda::CFieldEntryViewer<T, Layout, M, N>;
//    using Base::Base;
//    using Base::operator();
//    MUDA_GENERIC CMatrixEntryViewer(const Base& base)
//        : Base(base)
//    {
//    }
//
//    MUDA_GENERIC auto operator()(int i) const
//        -> Eigen::Map<const Eigen::Matrix<T, M, N>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
//};
//
//template <typename T, FieldEntryLayout Layout, int M, int N>
//class MatrixEntryViewer : public FieldEntryViewer<T, Layout, M, N>
//{
//  public:
//    using Base = muda::FieldEntryViewer<T, Layout, M, N>;
//    using Base::Base;
//    using Base::operator();
//    MUDA_GENERIC MatrixEntryViewer(const Base& base)
//        : Base(base)
//    {
//    }
//
//    MUDA_GENERIC auto operator()(int i)
//        -> Eigen::Map<Eigen::Matrix<T, M, N>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
//
//    MUDA_GENERIC operator CMatrixEntryViewer<T, Layout, M, N>() const
//    {
//        return CMatrixEntryViewer<T, Layout, M, N>(
//            Base::operator typename CMatrixEntryViewer<T, Layout, M, N>::Base());
//    }
//};
//
//template <typename T, FieldEntryLayout Layout, int N>
//using VectorEntryViewer = MatrixEntryViewer<T, Layout, N, 1>;
//template <typename T, FieldEntryLayout Layout, int N>
//using CVectorEntryViewer = CMatrixEntryViewer<T, Layout, N, 1>;
//template <typename T, FieldEntryLayout Layout>
//using Vector2EntryViewer = MatrixEntryViewer<T, Layout, 2, 1>;
//template <typename T, FieldEntryLayout Layout>
//using CVector2EntryViewer = CMatrixEntryViewer<T, Layout, 2, 1>;
//template <typename T, FieldEntryLayout Layout>
//using Vector3EntryViewer = MatrixEntryViewer<T, Layout, 3, 1>;
//template <typename T, FieldEntryLayout Layout>
//using CVector3EntryViewer = CMatrixEntryViewer<T, Layout, 3, 1>;
//template <typename T, FieldEntryLayout Layout>
//using Vector4EntryViewer = MatrixEntryViewer<T, Layout, 4, 1>;
//template <typename T, FieldEntryLayout Layout>
//using CVector4EntryViewer = CMatrixEntryViewer<T, Layout, 4, 1>;
//template <typename T, FieldEntryLayout Layout>
//using Matrix2x2EntryViewer = MatrixEntryViewer<T, Layout, 2, 2>;
//template <typename T, FieldEntryLayout Layout>
//using CMatrix2x2EntryViewer = CMatrixEntryViewer<T, Layout, 2, 2>;
//template <typename T, FieldEntryLayout Layout>
//using Matrix3x3EntryViewer = MatrixEntryViewer<T, Layout, 3, 3>;
//template <typename T, FieldEntryLayout Layout>
//using CMatrix3x3EntryViewer = CMatrixEntryViewer<T, Layout, 3, 3>;
//template <typename T, FieldEntryLayout Layout>
//using Matrix4x4EntryViewer = MatrixEntryViewer<T, Layout, 4, 4>;
//template <typename T, FieldEntryLayout Layout>
//using CMatrix4x4EntryViewer = CMatrixEntryViewer<T, Layout, 4, 4>;
//
//template <typename T, FieldEntryLayout Layout>
//MUDA_HOST auto make_viewer(FieldEntry<T, Layout, 1, 1>& e)
//    -> FieldEntryViewer<T, Layout, 1, 1>;
//template <typename T, FieldEntryLayout Layout>
//MUDA_HOST auto make_cviewer(FieldEntry<T, Layout, 1, 1>& e)
//    -> CFieldEntryViewer<T, Layout, 1, 1>;
//template <typename T, FieldEntryLayout Layout, int M, int N>
//MUDA_HOST auto make_viewer(FieldEntry<T, Layout, M, N>& e)
//    -> MatrixEntryViewer<T, Layout, M, N>;
//template <typename T, FieldEntryLayout Layout, int M, int N>
//MUDA_HOST auto make_cviewer(const FieldEntry<T, Layout, M, N>& e)
//    -> CMatrixEntryViewer<T, Layout, M, N>;
//}  // namespace muda::eigen
//
//namespace muda
//{
//template <typename T, FieldEntryLayout Layout, int M, int N>
//struct read_only_viewer<eigen::MatrixEntryViewer<T, Layout, M, N>>
//{
//    using type = eigen::CMatrixEntryViewer<T, Layout, M, N>;
//};
//
//template <typename T, FieldEntryLayout Layout, int M, int N>
//struct read_write_viewer<eigen::CMatrixEntryViewer<T, Layout, M, N>>
//{
//    using type = eigen::MatrixEntryViewer<T, Layout, M, N>;
//};
//}  // namespace muda
//
//#include "details/field_entry_viewer.inl"